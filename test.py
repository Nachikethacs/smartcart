import os
import csv
import random

f = open("prices.csv", "w", newline = "")
writer = csv.writer(f)
writer.writerow(['name', 'price'])
f.close()

f1 = open("prices.csv", "a", newline = "")
writer = csv.writer(f1)
for i in os.listdir("static/test_color"):
    writer.writerow([i.split('.')[0], str(random.randint(500, 600))])
f1.close()

print("done")