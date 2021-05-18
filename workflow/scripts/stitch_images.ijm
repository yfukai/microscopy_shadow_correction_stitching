arg = getArgument();
print(arg);
arg = split(arg, "@");
directory=arg[0];
layout_file=arg[1];

run("Grid/Collection stitching", 
	"type=[Positions from file] "+
	"order=[Defined by TileConfiguration] "+
	"directory="+directory+" "+
	"layout_file="+layout_file+" "+
	"fusion_method=[Do not fuse images (only write TileConfiguration)] "+
	"regression_threshold=0.30 "+
	"max/avg_displacement_threshold=2.50 "+
	"absolute_displacement_threshold=3.50 "+
	"compute_overlap "+
	"display_fusion "+
	"computation_parameters=[Save computation time (but use more RAM)] "+
	"image_output=[Fuse and display]");
