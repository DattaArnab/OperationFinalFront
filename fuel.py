# COMPLETE PIPELINE FOR OPERATION FINAL FRONT++
import cv2
import numpy as np
import heapq
from digit_classifier import predict_from_numpy
class GraphAdjacencyMatrixExtractor:
    def detect_nodes(self, image):
        """
        Detect circular nodes in the graph
        Returns node positions and radii
        """
        # Convert to grayscale if necessary
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # Use Hough Circle Transform to detect circles
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=10,
            param1=50,
            param2=30,
            minRadius=15,
            maxRadius=50
        )

        # If no circles are detected, try an alternative approach with blob detection
        if circles is None:
            # Set up SimpleBlobDetector parameters
            params = cv2.SimpleBlobDetector_Params()
            params.filterByCircularity = True
            params.minCircularity = 0.8
            params.filterByConvexity = True
            params.minConvexity = 0.9
            params.filterByInertia = True
            params.minInertiaRatio = 0.5

            detector = cv2.SimpleBlobDetector_create(params)
            keypoints = detector.detect(blurred)

            # Convert keypoints to circles format
            circles = np.array([[[kp.pt[0], kp.pt[1], kp.size / 2]] for kp in keypoints])

        return circles[0] if circles is not None else np.array([])

    def recognize_digits(self, image, nodes):
        """
        Extract and recognize digits inside the detected nodes using the model
        """

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Apply thresholding to create binary image for better text extraction
        # Use adaptive thresholding for better performance across different lighting conditions

        # Apply morphological operations to clean up the image
        kernel = np.ones((2, 2), np.uint8)

        labels = []
        confidences = []

        for x, y, r in nodes:
            # Extract a square ROI centered at the node center with side length = diameter
            x, y, r = int(x), int(y), int(r)

            # Calculate the square ROI bounds
            x_min = max(0, x - r)
            x_max = min(gray.shape[1], x + r)
            y_min = max(0, y - r)
            y_max = min(gray.shape[0], y + r)

            # Extract ROI
            roi = gray[y_min:y_max, x_min:x_max]

            if roi.size == 0:
                # If ROI extraction failed, assign None and continue
                labels.append(None)
                confidences.append(0.0)
                continue

            if roi is None:
                labels.append(None)
                confidences.append(0.0)
                continue

            # Use our digit classifier for prediction
            digit, confidence = predict_from_numpy(roi)

            # Higher confidence threshold (0.7) to ensure reliable predictions
            if digit is not None and confidence > 0.7:
                labels.append(digit)
                confidences.append(confidence)
            else:
                # If model confidence is too low, assign None
                # We don't want to fallback to arbitrary labels
                labels.append(None)
                confidences.append(confidence if confidence else 0.0)

        # Handle any None or duplicate labels in a more principled way
        self._post_process_labels(labels, confidences)

        return labels

    def _post_process_labels(self, labels, confidences):
        """
        Post-process labels to handle None values and duplicate labels
        with a principled approach that respects model predictions
        """
        if not labels:
            return

        # Generate indices for all nodes
        indices = list(range(len(labels)))

        # Sort indices by confidence (higher confidence first)
        sorted_indices = sorted(indices, key=lambda i: confidences[i] if confidences[i] is not None else 0,
                                reverse=True)

        # Keep track of assigned labels to avoid duplicates
        assigned_labels = set()

        # First pass - keep high confidence predictions and track used labels
        for i in sorted_indices:
            if labels[i] is not None and confidences[i] > 0.7:
                if labels[i] not in assigned_labels:
                    assigned_labels.add(labels[i])
                # else: we'll handle duplicates in the second pass

        # Second pass - resolve None values and duplicates
        for i in range(len(labels)):
            if labels[i] is None or (labels[i] in assigned_labels and confidences[i] <= 0.7):
                # Find an unused label for this node
                for label_candidate in range(len(labels)):
                    if label_candidate not in assigned_labels:
                        labels[i] = label_candidate
                        assigned_labels.add(label_candidate)
                        break
            else:
                # This is a unique label with high confidence, keep it
                assigned_labels.add(labels[i])

    def detect_edges_and_arrows(self, image, nodes):
        """
        Detect edges and their directions using cavity detection (primary)
        and confirm arrowheads with the strip-based method.
        Returns:
            edges: list of (start_idx, end_idx)
            directions: list of bool (True: start->end, False: end->start)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            color_img = image.copy()
        else:
            gray = image
            color_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Mask out nodes
        node_mask = np.zeros_like(gray)
        for x, y, r in nodes:
            cv2.circle(node_mask, (int(x), int(y)), int(r) - 2, 255, -1)
        edge_mask = cv2.bitwise_not(node_mask)

        # Adaptive Canny + dilation
        v = np.median(gray)
        lower = int(max(0, 0.66 * v))
        upper = int(min(255, 1.33 * v))
        edges = cv2.Canny(gray, lower, upper)
        edges = cv2.bitwise_and(edges, edges, mask=edge_mask)
        kernel = np.ones((3, 3), np.uint8)
        dilated_edges = cv2.dilate(edges, kernel, iterations=1)

        n = len(nodes)
        edge_list = []
        direction_list = []

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                x1, y1, r1 = nodes[i]
                x2, y2, r2 = nodes[j]
                dx, dy = x2 - x1, y2 - y1
                dist = np.hypot(dx, dy)
                if dist < 1e-3:
                    continue
                start_x = int(x1 + dx * r1 / dist)
                start_y = int(y1 + dy * r1 / dist)
                end_x = int(x2 - dx * r2 / dist)
                end_y = int(y2 - dy * r2 / dist)
                width = max(6, int((r1 + r2) * 0.2))
                rect_len = int(np.hypot(end_x - start_x, end_y - start_y))
                if rect_len < 5:
                    continue
                angle = np.arctan2(end_y - start_y, end_x - start_x)
                wx = np.sin(angle) * width / 2
                wy = np.cos(angle) * width / 2
                pts = np.array([
                    [start_x - wx, start_y + wy],
                    [start_x + wx, start_y - wy],
                    [end_x + wx, end_y - wy],
                    [end_x - wx, end_y + wy]
                ], dtype=np.int32)
                rect_mask = np.zeros_like(gray)
                cv2.fillPoly(rect_mask, [pts], 255)
                total_rect = np.sum(rect_mask > 0)
                lit_rect = np.sum((dilated_edges > 0) & (rect_mask > 0))
                thresh = 0.4
                if rect_len < 60:
                    threshold = thresh * total_rect
                elif rect_len < 180:
                    threshold = thresh * total_rect - (r1 + r2) * width
                else:
                    threshold = thresh * total_rect - (r1 + r2) * width * 2
                if lit_rect >= threshold:
                    start_circle = np.zeros_like(gray, dtype=np.uint8)
                    end_circle = np.zeros_like(gray, dtype=np.uint8)
                    cv2.circle(start_circle, (int(x1), int(y1)), int(r1 * 1.5), 255, -1)
                    cv2.circle(end_circle, (int(x2), int(y2)), int(r2 * 1.5), 255, -1)
                    start_mask = cv2.bitwise_and(rect_mask, start_circle)
                    end_mask = cv2.bitwise_and(rect_mask, end_circle)

                    # --- Cavity detection (primary) ---
                    close_kernel = np.ones((5, 5), np.uint8)
                    # Start
                    start_region_edges = cv2.bitwise_and(dilated_edges, dilated_edges, mask=start_mask)
                    start_filled = cv2.morphologyEx(start_region_edges, cv2.MORPH_CLOSE, close_kernel)
                    start_cavity_pixels = np.sum(start_filled > 0) - np.sum(start_region_edges > 0)
                    start_cavity = start_cavity_pixels > 1
                    # End
                    end_region_edges = cv2.bitwise_and(dilated_edges, dilated_edges, mask=end_mask)
                    end_filled = cv2.morphologyEx(end_region_edges, cv2.MORPH_CLOSE, close_kernel)
                    end_cavity_pixels = np.sum(end_filled > 0) - np.sum(end_region_edges > 0)
                    end_cavity = end_cavity_pixels > 1
                    if (not start_cavity and not end_cavity):
                        start_cavity = start_cavity_pixels >= 1
                        end_cavity = end_cavity_pixels >= 1
                    # --- Strip-based confirmation (secondary, for each arrowhead) ---
                    strip_width = 3
                    strip_threshold = 0.4  # at most 90% lit pixels to avoid false positive
                    dir_vec = np.array([dx, dy]) / dist
                    perp_vec = np.array([-dir_vec[1], dir_vec[0]])
                    strip_len = width
                    strip_brightness_thresh = 220

                    # For start
                    start_left_strip = np.zeros_like(gray, dtype=np.uint8)
                    start_left_center = np.array([start_x, start_y]) - perp_vec * (width / 2)
                    start_left_pts = np.array([
                        start_left_center,
                        start_left_center + perp_vec * strip_width,
                        start_left_center + perp_vec * strip_width + dir_vec * strip_len,
                        start_left_center + dir_vec * strip_len
                    ], dtype=np.int32)
                    cv2.fillPoly(start_left_strip, [start_left_pts], 255)
                    start_left_strip = cv2.bitwise_and(start_left_strip, start_mask)

                    start_right_strip = np.zeros_like(gray, dtype=np.uint8)
                    start_right_center = np.array([start_x, start_y]) + perp_vec * (width / 2)
                    start_right_pts = np.array([
                        start_right_center,
                        start_right_center - perp_vec * strip_width,
                        start_right_center - perp_vec * strip_width + dir_vec * strip_len,
                        start_right_center + dir_vec * strip_len
                    ], dtype=np.int32)
                    cv2.fillPoly(start_right_strip, [start_right_pts], 255)
                    start_right_strip = cv2.bitwise_and(start_right_strip, start_mask)

                    start_left_total = np.sum(start_left_strip > 0)
                    start_left_lit = np.sum((gray > strip_brightness_thresh) & (start_left_strip > 0))
                    start_left_ratio = start_left_lit / start_left_total if start_left_total > 0 else 0

                    start_right_total = np.sum(start_right_strip > 0)
                    start_right_lit = np.sum((gray > strip_brightness_thresh) & (start_right_strip > 0))
                    start_right_ratio = start_right_lit / start_right_total if start_right_total > 0 else 0

                    start_cavity_confirmed = start_cavity and (
                                start_left_ratio <= strip_threshold or start_right_ratio <= strip_threshold)

                    # For end
                    end_left_strip = np.zeros_like(gray, dtype=np.uint8)
                    end_left_center = np.array([end_x, end_y]) - perp_vec * (width / 2)
                    end_left_pts = np.array([
                        end_left_center,
                        end_left_center + perp_vec * strip_width,
                        end_left_center + perp_vec * strip_width - dir_vec * strip_len,
                        end_left_center - dir_vec * strip_len
                    ], dtype=np.int32)
                    cv2.fillPoly(end_left_strip, [end_left_pts], 255)
                    end_left_strip = cv2.bitwise_and(end_left_strip, end_mask)

                    end_right_strip = np.zeros_like(gray, dtype=np.uint8)
                    end_right_center = np.array([end_x, end_y]) + perp_vec * (width / 2)
                    end_right_pts = np.array([
                        end_right_center,
                        end_right_center - perp_vec * strip_width,
                        end_right_center - perp_vec * strip_width - dir_vec * strip_len,
                        end_right_center - dir_vec * strip_len
                    ], dtype=np.int32)
                    cv2.fillPoly(end_right_strip, [end_right_pts], 255)
                    end_right_strip = cv2.bitwise_and(end_right_strip, end_mask)

                    end_left_total = np.sum(end_left_strip > 0)
                    end_left_lit = np.sum((gray > strip_brightness_thresh) & (end_left_strip > 0))
                    end_left_ratio = end_left_lit / end_left_total if end_left_total > 0 else 0

                    end_right_total = np.sum(end_right_strip > 0)
                    end_right_lit = np.sum((gray > strip_brightness_thresh) & (end_right_strip > 0))
                    end_right_ratio = end_right_lit / end_right_total if end_right_total > 0 else 0

                    end_cavity_confirmed = end_cavity and (
                                end_left_ratio <= strip_threshold or end_right_ratio <= strip_threshold)

                    # Accept edge if either end is confirmed as an arrowhead (even both)
                    if end_cavity_confirmed:
                        edge_list.append((i, j))
                        direction_list.append(True)
                    if start_cavity_confirmed:
                        edge_list.append((j, i))
                        direction_list.append(True)
        return edge_list, direction_list

    def preprocess_image(self, image):
        """
        Resize image to 419x440 and stretch intensity so min=0, max=255.
        """
        image = cv2.resize(image, (419, 440), interpolation=cv2.INTER_AREA)
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        min_val = np.min(gray)
        max_val = np.max(gray)
        if max_val > min_val:
            norm = ((gray - min_val) * (255.0 / (max_val - min_val))).astype(np.uint8)
        else:
            norm = np.zeros_like(gray)
        return norm
    def extract_adjacency_matrix(self, image_path):
        """
        Main function to extract adjacency matrix from a graph image
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        # Preprocess image
        image = self.preprocess_image(image)
        # Detect nodes
        nodes = self.detect_nodes(image)
        if len(nodes) == 0:
            return np.array([]), []
        # Recognize digits
        labels = self.recognize_digits(image, nodes)
        # Detect edges and arrow directions together
        edges, directions = self.detect_edges_and_arrows(image, nodes)
        # Create adjacency matrix
        n = len(nodes)
        adjacency_matrix = np.zeros((n, n), dtype=int)
        for (start_idx, end_idx), direction in zip(edges, directions):
            adjacency_matrix[start_idx, end_idx] = 1
        return adjacency_matrix, labels

    def order_adjacency_matrix(self, adjacency_matrix, labels):
        """
        Reorder the adjacency matrix based on the node labels
        Returns:
            ordered_matrix: Reordered adjacency matrix
            ordered_labels: Labels in increasing order
        """
        # Create mapping from current indices to label values
        label_indices = [(i, label) for i, label in enumerate(labels)]
        # Sort by label value
        label_indices.sort(key=lambda x: x[1])
        # Create mapping from old indices to new indices
        old_to_new = {old_idx: new_idx for new_idx, (old_idx, _) in enumerate(label_indices)}
        # Get ordered labels
        ordered_labels = [label for _, label in label_indices]
        # Create the reordered adjacency matrix
        n = len(adjacency_matrix)
        ordered_matrix = np.zeros((n, n), dtype=int)
        for i in range(n):
            for j in range(n):
                if adjacency_matrix[i, j] == 1:
                    new_i = old_to_new[i]
                    new_j = old_to_new[j]
                    ordered_matrix[new_i, new_j] = 1
        return ordered_matrix, ordered_labels
def solve_minimum_fuel_optimized(adjacency_matrix, start_node, end_node):
    """
    Optimized solution using Dijkstra's algorithm for the Operation Final Front++ problem.
    State space: (node, flip_state) where flip_state indicates if signals are reversed.
    Operations:
    1. Traverse edge: cost = 1
    2. Reverse all signals: cost = N (number of nodes)
    Parameters:
    - adjacency_matrix: 2D numpy array representing the directed graph
    - start_node: Starting node index
    - end_node: Target node index
    Returns:
    - Minimum fuel required or -1 if impossible
    """
    n = len(adjacency_matrix)
    if start_node < 0 or start_node >= n or end_node < 0 or end_node >= n:
        return -1
    if not isinstance(adjacency_matrix, np.ndarray):
        adjacency_matrix = np.array(adjacency_matrix)
    pq = [(0, start_node, 0)]  # (fuel_cost, node, flip_state)
    dist = {(start_node, 0): 0}
    while pq:
        fuel, node, flip_state = heapq.heappop(pq)
        if (node, flip_state) in dist and dist[(node, flip_state)] < fuel:
            continue
        if node == end_node:
            return fuel
        current_adj = adjacency_matrix if flip_state == 0 else adjacency_matrix.T
        for next_node in range(n):
            if current_adj[node][next_node] == 1:
                new_fuel = fuel + 1
                state = (next_node, flip_state)
                if state not in dist or new_fuel < dist[state]:
                    dist[state] = new_fuel
                    heapq.heappush(pq, (new_fuel, next_node, flip_state))
        # Reverse all signals
        new_flip_state = 1 - flip_state
        new_fuel = fuel + n
        state = (node, new_flip_state)
        if state not in dist or new_fuel < dist[state]:
            dist[state] = new_fuel
            heapq.heappush(pq, (new_fuel, node, new_flip_state))
    return -1


def main(image_path, start_label=0, end_label=None):
    """
    Process a graph image and extract its adjacency matrix,
    then solve the Operation Final Front++ problem
    """
    extractor = GraphAdjacencyMatrixExtractor()
    try:
        # Extract adjacency matrix
        adjacency_matrix, labels = extractor.extract_adjacency_matrix(image_path)
        # Get ordered matrix for visualization
        ordered_matrix, ordered_labels = extractor.order_adjacency_matrix(adjacency_matrix, labels)
        # If labels are not integers, convert them to indices
        label_to_idx = {label: i for i, label in enumerate(labels)}
        # If we found a node labeled specifically as the start_label
        if start_label in labels:
            start_idx = labels.index(start_label)
        else:
            # Otherwise default to the first node
            start_idx = 0
        # If end_label is not specified, use the highest numbered node as target
        if end_label is None:
            numeric_labels = [label for label in labels if isinstance(label, int)]
            end_label = max(numeric_labels) if numeric_labels else len(labels) - 1
        # If we found a node labeled specifically as the end_label
        if end_label in labels:
            end_idx = labels.index(end_label)
        else:
            # Otherwise default to the last node
            end_idx = len(labels) - 1
        # Solve the minimum fuel problem (Dijkstra optimized)
        min_fuel_optimized = solve_minimum_fuel_optimized(adjacency_matrix, start_idx, end_idx)
        print(f"Minimum fuel : {min_fuel_optimized}")
        return adjacency_matrix, labels, min_fuel_optimized, ordered_matrix, ordered_labels
    except Exception as e:
        print(f"Error: {e}")
        return None, None, -1, None, None


if __name__ == "__main__":
    import sys
    # Default image path
    default_image_path = "graphs_images/1.png"
    # Get image path from command line arguments if provided
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        print(f"Using image from command line argument: {image_path}")
    else:
        image_path = default_image_path
        print(f"No image path provided, using default: {default_image_path}")
    # Optional: Allow start_label and end_label as additional arguments
    start_label = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    end_label = int(sys.argv[3]) if len(sys.argv) > 3 else None
    # Run main function with the provided arguments
    main(image_path, start_label, end_label)
