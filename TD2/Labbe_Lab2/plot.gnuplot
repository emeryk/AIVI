set terminal png
set output "outMulti.png"

set xlabel "Frames"
set ylabel "MSE"
#set xrange [0:110]
#set yrange [0: 700]
set xtics 50
set ytics 50
set style line 1 lw 5
set style line 2 lw 5
set style line 3 lw 5
plot 'stats.txt' using 1:2 with lines title 'MSE 1', 'stats.txt' using 1:3 with lines title 'MSE 2', 'stats.txt' using 1:4 with lines title 'MSE 3'
