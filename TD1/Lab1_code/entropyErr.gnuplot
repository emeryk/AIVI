set terminal png
set output "outErr.png"

set xlabel "Frames"
set ylabel "ENTROPY, ERROR"
#set xrange [0:110]
#set yrange [0: 700]
set xtics 10
set ytics 0.2
set style line 1 lw 5
set style line 2 lw 5
plot 'statsEntropy.txt' using 1:2 with lines title 'ENTROPY', 'statsEntropy.txt' using 1:3 with lines title 'ERROR'
