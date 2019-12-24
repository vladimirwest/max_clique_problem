# max_clique_problem

Instance|Time (s)|Max clique len|Max clique verticies
---|---|---|---|
brock200_2|1743|12|27,48,55,70,105,120,121,135,145,149,158,183
brock200_3|10560|15|11,28,35,37,57,83,96,97,103,117,129,143,157,172,177
brock200_4|43200|17|12,19,28,29,38,54,65,71,79,93,117,127,139,161,165,186,192
keller4|5460|11|2,6,22,24,74,76,100,109,118,140,153
MANN_a9|156|16|3,4,5,9,10,15,16,19,24,27,28,33,36,38,42,45
hamming6-2.clq|5
hamming6-4.clq|2
johnson8-2-4.clq|0.5
johnson8-4-4.clq|54
johnson16-2-4.clq|8105
p_hat300-1.clq|1084|8|54,124,181,219,247,268,271,287
p_hat500-1.clq|12420

# Cplex solution

Instance|Time (ms)|Max clique len|Max clique verticies
---|---|---|---|
C125.9| 9686|34|7,9,11,13,19,22,25,29,33,34,40,44,49,52,54,55,66,67,68,70,79,80,93,96,98,99,103,104,110,111,114,117,122,125
MANN_a9| 1615|16|3,4,5,9,10,15,16,19,24,27,28,33,36,38,42,45
hamming6-2| ~0 (initial optimal value found by cplex is already integer)
johnson8-2-4| 1751
johnson8-4-4| 3183
