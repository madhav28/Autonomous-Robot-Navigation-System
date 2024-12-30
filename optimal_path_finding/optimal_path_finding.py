import cv2
import numpy as np
import heapq
from scipy.ndimage import distance_transform_edt

# Load image and create red mask
image = cv2.imread('birdseye_view_binary.png')
red_mask = (image[:, :, 2] > 200) & (image[:, :, 1] < 100) & (image[:, :, 0] < 100)

# Create binary map (1 for obstacles, 0 for traversable)
binary_map = np.ones_like(red_mask, dtype=int)
binary_map[~red_mask] = 0

# Precompute distances to red regions
distance_to_red = distance_transform_edt(~red_mask)

# Parameters
# min_distance = 6
# start = (307, 321)
# end = (746, 564)
# start_object = 'table'
# end_object = 'bed'

# min_distance = 6
# start = (241, 602)
# end = (312, 359)
# start_object = 'microwave'
# end_object = 'couch'

# min_distance = 6
# start = (378, 138)
# end = (644, 769)
# start_object = 'tv'
# end_object = 'sink'

min_distance = 6
start = (211, 587)
end = (770, 287)
start_object = 'toaster'
end_object = 'chair'

# Heuristic function (Euclidean distance)
def heuristic(a, b):
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

# Get neighbors within bounds
def get_neighbors(current, shape):
    x, y = current
    neighbors = [
        (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1),
        (x + 1, y + 1), (x + 1, y - 1), (x - 1, y + 1), (x - 1, y - 1)
    ]
    return [n for n in neighbors if 0 <= n[0] < shape[0] and 0 <= n[1] < shape[1]]

# Reconstruct path from the came_from map
def reconstruct_path(came_from, current):
    path = []
    while current in came_from:
        path.append(current)
        current = came_from[current]
    path.append(current)
    return path[::-1]

# A* algorithm
def a_star(start, end, grid, distance_to_red, min_distance):
    open_list = []
    heapq.heappush(open_list, (0 + heuristic(start, end), 0, start))
    came_from = {}
    g_cost = {start: 0}
    while open_list:
        _, current_g, current = heapq.heappop(open_list)
        if current == end:
            return reconstruct_path(came_from, current)
        
        for neighbor in get_neighbors(current, grid.shape):
            if grid[neighbor] == 1:  # Obstacle
                continue
            if distance_to_red[neighbor] < min_distance:  # Too close to red
                continue

            # Compute cost
            tentative_g = current_g + (1.414 if neighbor[0] != current[0] and neighbor[1] != current[1] else 1)
            if neighbor not in g_cost or tentative_g < g_cost[neighbor]:
                g_cost[neighbor] = tentative_g
                f_cost = tentative_g + heuristic(neighbor, end)
                heapq.heappush(open_list, (f_cost, tentative_g, neighbor))
                came_from[neighbor] = current
    return None

# Find optimal path
path = a_star(start, end, binary_map, distance_to_red, min_distance)

# Annotate start and end points
cv2.circle(image, (start[1], start[0]), radius=5, color=(255, 255, 255), thickness=-1) 
cv2.putText(image, "Start from "+start_object, (start[1] + 10, start[0] + 10), 
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

cv2.circle(image, (end[1], end[0]), radius=5, color=(255, 255, 255), thickness=-1) 
cv2.putText(image, "End to "+end_object, (end[1] + 10, end[0] + 10), 
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

# Draw path on the image with thickness
if path:
    for i in range(len(path) - 1):
        start_point = (path[i][1], path[i][0])  # (x, y)
        end_point = (path[i + 1][1], path[i + 1][0])  # (x, y)
        cv2.line(image, start_point, end_point, color=(255, 255, 255), thickness=3)

cv2.imwrite('optimal_path_'+start_object+'_to_'+end_object+'.png', image)