#!/usr/bin/gnuplot -persist

set xrange [0:101]
set yrange [0:101]
set ticslevel 0
splot "expectation_difusion_x100.dat" u 1:2:3 with lines, \
      "simulation_difusion_x100.dat" u 1:2:3 with lines

pause -1
