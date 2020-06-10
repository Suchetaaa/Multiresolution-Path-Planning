"""
Graph generation, path planning using Dijkstra's algorithm

Author: Aniket Bhatia
Date: 08 June, 2020
"""
# Import libraries
import numpy as np
from dijkstar import Graph, find_path
import cv2
from mra import cellDecomp


def make_graph(image, info, visited):
    graph = Graph()
    m, n, _ = np.shape(image)

    for i in range(m):
        for j in range(n):
            node_val = image[i][j][1]
            if(i != 0):
                node_val_left = image[i - 1][j][1]
                if(node_val != node_val_left):
                    alt_diff = 0
                    if(alt_diff < 0):
                        alt_diff = 0
                    euclid_dist = cell_dist(node_val, node_val_left, info)
                    alt_area = image[i - 1][j][0] * \
                        new_area(node_val_left, info)
                    temp = alt_diff + euclid_dist + alt_area

                    weight = temp
                    graph.add_edge(node_val, node_val_left, weight)

            if(i != m - 1):
                node_val_right = image[i + 1][j][1]
                if(node_val != node_val_right):

                    alt_diff = 0
                    if(alt_diff < 0):
                        alt_diff = 0
                    euclid_dist = cell_dist(node_val, node_val_right, info)
                    alt_area = image[i + 1][j][0] * \
                        new_area(node_val_right, info)
                    temp = alt_diff + euclid_dist + alt_area

                    weight = temp
                    graph.add_edge(node_val, node_val_right, weight)

            if(j != 0):
                node_val_up = image[i][j - 1][1]
                if(node_val != node_val_up):

                    alt_diff = 0
                    if(alt_diff < 0):
                        alt_diff = 0
                    euclid_dist = cell_dist(node_val, node_val_up, info)
                    alt_area = image[i][j - 1][0] * new_area(node_val_up, info)
                    temp = alt_diff + euclid_dist + alt_area

                    weight = temp
                    graph.add_edge(node_val, node_val_up, weight)

            if(j != n - 1):
                node_val_down = image[i][j + 1][1]
                if(node_val != node_val_down):

                    alt_diff = 0
                    if(alt_diff < 0):
                        alt_diff = 0
                    euclid_dist = cell_dist(node_val, node_val_down, info)
                    alt_area = image[i][j + 1][0] * \
                        new_area(node_val_down, info)
                    temp = alt_diff + euclid_dist + alt_area

                    weight = temp
                    graph.add_edge(node_val, node_val_down, weight)

    return graph


def get_path(graph, src, dest, image):

    source_node = image[src[0]][src[1]][1]
    dest_node = image[dest[0]][dest[1]][1]

    path = find_path(graph, source_node, dest_node)

    return path


def dist(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def cell_dist(n1, n2, info):

    flag1 = False
    flag2 = False
    for node in info.keys():
        if(node == n1):
            tup1 = info[node]
            flag1 = True
        elif(node == n2):
            tup2 = info[node]
            flag2 = True

        if(flag1 and flag2):
            break
    if(flag2 == False):
        print("n1:", n1, "n2:", n2)
    distance = dist((tup1[1], tup1[0]), (tup2[1], tup2[0]))
    return distance


def new_area(node, info):
    for n in info:
        if(n == node):
            return info[n][2]


def node_to_point(node, image):
    m, n, _ = np.shape(image)

    for i in range(m):
        for j in range(n):
            if(image[i][j][1] == node):
                return [i, j]


def path_plan(img, src, dest, thresh):

    # thresh = 55
    next_pt = src
    visited = []
    iter = 0
    while(dist(next_pt, dest) > thresh):
        image, info = cellDecomp(img, next_pt[0], next_pt[1])
        iter += 1
        print("iter:", iter, "current_location",
              next_pt, "dist:", dist(next_pt, dest))

        graph = make_graph(image, info, visited)
        if(len(visited) >= 2):
            node_now = image[next_pt[0]][next_pt[1]][1]
            for prev in visited:
                if(dist(next_pt, prev) == 1):
                    node_prev = image[prev[0]][prev[1]][1]
                    graph.add_edge(node_now, node_prev, float('inf'))

        path = get_path(graph, next_pt, dest, image)
        next_node = path.nodes[1]
        next_pt = node_to_point(next_node, image)
        # next_pt = np.array(location)

        visited.append(next_pt)

    if(next_pt != dest):
        visited.append(dest)

    return visited


if __name__ == '__main__':

    img = cv2.imread('frag2.PNG', cv2.IMREAD_GRAYSCALE)
    img_c = cv2.imread('frag2.png')

    img = cv2.resize(img, (256, 256))
    img_c = cv2.resize(img_c, (256, 256))

    img = 255 - img
    vis = path_plan(img, (170, 180), (81, 93), 0.5)

    img_c[170][180][0] = 0
    img_c[170][180][1] = 0
    img_c[170][180][2] = 255

    for p in vis:
        for i in range(3):
            img_c[p[0]][p[1]][i] = 0

    cv2.imwrite('output.png', img_c)
