import compas_cgal.straight_skeleton_2 as ss2

# Define outer boundary (anticlockwise logic, normal [0,0,1])
points = [
    [0.0, 0.0, 0.0],
    [10.0, 0.0, 0.0],
    [10.0, 10.0, 0.0],
    [0.0, 10.0, 0.0]
]

# Define hole (clockwise logic relative to outer, normal should be [0,0,-1])
# We will define it as anticlockwise, compas_cgal straight_skeleton_2 handles reversing if needed
holes = [
    [
        [2.0, 2.0, 0.0],
        [8.0, 2.0, 0.0],
        [8.0, 8.0, 0.0],
        [2.0, 8.0, 0.0]
    ]
]

print("=== Testing Inner Offset ===")
weights_inner = [1.0, 2.0, 1.0, 2.0]
holes_weights_inner = [[1.0, 0.5, 1.0, 0.5]]
try:
    res_inner = ss2.weighted_offset_polygon_with_holes_inner(points, holes, 0.5, weights_inner, holes_weights=holes_weights_inner)
    print("Inner offset created successfully.")
    for outer, inner_holes in res_inner:
        print("Outer polygon size:", len(outer.points))
        for h in inner_holes:
            print("Hole polygon size:", len(h.points))
except Exception as e:
    print(f"Error inner: {e}")

print("=== Testing Outer Offset ===")
weights_outer = [1.5, 1.5, 1.5, 1.5]
holes_weights_outer = [[0.5, 0.5, 0.5, 0.5]]
try:
    res_outer = ss2.weighted_offset_polygon_with_holes_outer(points, holes, 0.5, weights_outer, holes_weights=holes_weights_outer)
    print("Outer offset created successfully.")
    for outer, inner_holes in res_outer:
        print("Outer polygon size:", len(outer.points))
        for h in inner_holes:
            print("Hole polygon size:", len(h.points))
except Exception as e:
    print(f"Error outer: {e}")
