// Axisymmetric (r,z) half plane, r in [0,R], z in [-R,R], graded geometrically toward the origin.
// Target size h(rho) = min(hmax, max(hmin, c*rho)), rho = |(r,z)|.
// Override any parameter: gmsh ns_axisym.geo -2 -setnumber c 0.05 -setnumber hmin 1e-3 ...
DefineConstant[
  R     = {8,     Name "Parameters/R"},
  c     = {0.05,  Name "Parameters/c"},
  hmin  = {0.05*0.5*Sqrt(1e-3), Name "Parameters/hmin"},
  hmax  = {0.5,   Name "Parameters/hmax"},
  order = {1,     Name "Parameters/order"}
];

SetFactory("Built-in");

Point(1) = {0, -R, 0, hmax};
Point(2) = {R, -R, 0, hmax};
Point(3) = {R,  R, 0, hmax};
Point(4) = {0,  R, 0, hmax};
Point(5) = {0,  0, 0, hmin};   // origin: a mesh vertex sits at the singularity

Line(1) = {1, 2};   // bottom
Line(2) = {2, 3};   // outer
Line(3) = {3, 4};   // top
Line(4) = {4, 5};   // axis, upper half
Line(5) = {5, 1};   // axis, lower half

Curve Loop(1) = {1, 2, 3, 4, 5};
Plane Surface(1) = {1};

Physical Curve("axis")   = {4, 5};
Physical Curve("outer")  = {2};
Physical Curve("top")    = {3};
Physical Curve("bottom") = {1};
Physical Surface("fluid") = {1};

// Linear ramp between DistMin and DistMax reproduces c*rho exactly, clamped to [hmin, hmax].
Field[1] = Distance;
Field[1].PointsList = {5};
Field[2] = Threshold;
Field[2].InField  = 1;
Field[2].SizeMin  = hmin;
Field[2].SizeMax  = hmax;
Field[2].DistMin  = hmin / c;
Field[2].DistMax  = hmax / c;
Field[2].Sigmoid  = 0;
Field[3] = Min;
Field[3].FieldsList = {2};
Background Field = 3;

Mesh.MeshSizeFromPoints = 0;
Mesh.MeshSizeFromCurvature = 0;
Mesh.MeshSizeExtendFromBoundary = 0;
Mesh.Algorithm = 6;          // Frontal-Delaunay
Mesh.ElementOrder = order;
Mesh.MshFileVersion = 4.1;
