// The README showcase geometry: a 2 × 1 plate with two round holes, meshed by Gmsh when
// FerriteGmsh.togrid opens this file. Physical curves name the boundary pieces: "inlet" is the
// larger hole (a heated pipe), "outlet" the right edge and "wall" everything else.
SetFactory("OpenCASCADE");
Mesh.MeshSizeMax = 0.06;

Rectangle(1) = {0, 0, 0, 2, 1};
Disk(2) = {0.65, 0.55, 0, 0.22};
Disk(3) = {1.35, 0.4, 0, 0.18};
BooleanDifference(4) = {Surface{1}; Delete;}{Surface{2, 3}; Delete;};

eps = 1e-6;
inlet[] = Curve In BoundingBox{0.43 - eps, 0.33 - eps, -eps, 0.87 + eps, 0.77 + eps, eps};
outlet[] = Curve In BoundingBox{2 - eps, -eps, -eps, 2 + eps, 1 + eps, eps};
wall[] = Abs(Boundary{Surface{4};});
wall[] -= inlet[];
wall[] -= outlet[];

Physical Curve("inlet") = inlet[];
Physical Curve("outlet") = outlet[];
Physical Curve("wall") = wall[];
Physical Surface("domain") = {4};
