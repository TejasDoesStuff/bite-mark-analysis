#Convert 3 bite force trials into bite force (N)

values = []
for i in range(3):
  values.append(input("Enter value: " ))

avg = 0
for val in values:
  avg += int(val)
avg = avg/3
print("Average (Arduino):", round(avg));

#returns a small value because its detecting the force at a single point
#.6612 and 227.93 come from creating a linear equation between the bite force range and the expected teen range
force = 0.6612 * avg + 227.93
print("Bite Force (N):", round(force))