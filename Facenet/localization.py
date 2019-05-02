import csv
import pandas as pd
import numpy as np

# Observations and Assumptions:
# 1. the value of each one of the 512 dimensions is independent, and has the same distribution.
# 2. this distribution is independent of the label(person name).
# 3. it is a symmetry distribution, its mean value is 0 and standard deviation is 0.044
# 4. probability that a value is greater than standard deviation is 0.18.
# 5. the value of the 512 dimensional vector of different people, are independent.

# If the value of two 512 dimensional vector of the same person, are independent.
# Then, we can calculate:
# The probability that the two 512-dimensional vectors have 3 identical positions with a value
# greater than 0.044 is less than 0.5%.
# The probability that the two 512-dimensional vectors have 5 identical positions with a value
# greater than 0.044 is less than one in ten thousand.

# But, in fact, this value is greater than 0.3, typically larger than 0.4.
# This means the value of two 512 dimensional vector of the same person, are not independent.

# Therefore, we can group by the position of the dimensions greater than 0.044.

# Specifically,
# 1. Divide the 512 dimensions into num_of_group groups,
# each group containing num_of_dim_in_each_group dimensions.
# 2. Each 512-dimensional vector belongs to two groups:
# a group with the largest number of dimensions greater than 0.044,
# and a group with the second largest number of dimensions greater than 0.044.


num_of_group = 32
num_of_dim_in_each_group = int(512/num_of_group)
with open("./Client_Character_6000.csv","r") as client_data:
    data = csv.reader(client_data)
    with open("./Character_localization_6000.csv", "w") as localized_data:
        writer = csv.writer(localized_data)
        for line in data:
            # print('LINE', line)
            new_data = []
            new_data.append(line[0])
            loc = np.zeros(512)
            a = np.zeros(num_of_group)  # for each group 'a' stores the number of dimensions greater than 0.044
            for i in range(num_of_group):  # 32 groups
                for j in range(num_of_dim_in_each_group):  # 512/32=16 dimensoins in each group
                    # print('AAAA',line[2 + j + 32*i])
                    if float(line[1 + j + num_of_dim_in_each_group * i]) > 0.044:
                        a[i] += 1
                        loc[j + num_of_dim_in_each_group * i] = 1
                new_data.append(int(a[i]))
            new_data.append(str(loc).replace(' ', '').replace('.', ',').replace('[', '')
                            .replace(']', '').replace('\n', ''))
            print('NEW_DATA: ', new_data)
            writer.writerow(new_data)

with open("./Character_localization_6000.csv", "r") as localized_data:
    data = csv.reader(localized_data)
    with open("./Group_data_6000_statistic_method.txt", "w") as group_data:
        # writer = csv.writer(group_data, delimiter=';')
        group = [[x]for x in range(num_of_group)]
        j = 0
        for line in data:
            a = line[1:num_of_group+1]
            a = np.array(a)
            max = np.argmax(a)
            a[max] = 0
            sec_max = np.argmax(a)
            # a[sec_max] = 0
            # third_max = np.argmax(a)
            for i in range(num_of_group):
                if max == i or sec_max == i:  # or third_max == i:
                    group[i].append(j)  # group[i].append(line[0])
            j += 1
        for i in range(num_of_group):
            group_data.write(str(group[i]).replace('[', '').replace(']', ''))
            group_data.write(',')
            group_data.write('\n')
"""
with open("./Group_data.txt", "r") as location_data:
    while('True'):
        line = location_data.readline()
        print(np.shape(line.split(',')))
        if line.startswith('5,'):
            print(line,'AAAAA')
            num = 0
            a = 'a'
            while num < np.shape(line.split(','))[0]:
                num += 1
                a = line.split(',')[num-1].strip()
                print(num,a)
                if a == '\n':
                    break
        if(line == ''):
            break
####
with open("./Client_Character_dong.csv","r") as client_data:
    data = csv.reader(client_data)
    with open("./Character_localization_dong.csv", "w") as localized_data:
        writer = csv.writer(localized_data)
        for line in data:
            print('LINE', line)
            new_data = []
            new_data.append(line[0])
            point = []
            group_name = 0
            for i in range(2):
                if float(line[i+2]) < 0.01 and float(line[i+2]) > -0.01:
                    point.append(0)
                    group_name += (3 ** i) * 1
                elif float(line[i+2]) > 0:
                    point.append(1)
                    group_name += (3 ** i) * 2
                elif float(line[i+2]) < 0:
                    point.append(-1)
                    group_name += (3 ** i) * 0
            new_data.append(group_name)
            new_data.append(point)
            print('NEW_DATA: ', new_data)
            writer.writerow(new_data)

with open("./Character_localization_4.csv", "r") as localized_data:
    with open("./Group_data.csv", "w") as group_data:
        writer = csv.writer(group_data)
        for i in range(81):
            data = csv.reader(localized_data)
            print('i',i)
            client_in_this_group = []
            client_in_this_group.append(i)
            j = 0
            for line in data:
                print(line,j)
                print('AAA',line[1],i)
                if int(line[1])==i:
                    client_in_this_group.append(j)
                j += 1
            writer.writerow(client_in_this_group)
        print('Fin')

with open("./Character_localization_dong.csv", "r") as localized_data:
    data = csv.reader(localized_data)
    with open("./Group_data_dong.txt", "w") as group_data:
        # writer = csv.writer(group_data, delimiter=';')
        group = [[x]for x in range(3**2)]
        j = 0
        for line in data:
            for i in range(3**2):
                if int(line[1]) == i:
                    group[i].append(j)
            j += 1
        for i in range(3**2):
            group_data.write(str(group[i]).replace('[','').replace(']',''))
            group_data.write(',')
            group_data.write('\n')

with open("./Group_data.txt", "r") as location_data:
    while('True'):
        line = location_data.readline()
        print(np.shape(line.split(',')))
        if line.startswith('5,'):
            print(line,'AAAAA')
            num = 0
            a = 'a'
            while num < np.shape(line.split(','))[0]:
                num += 1
                a = line.split(',')[num-1].strip()
                print(num,a)
                if a == '\n':
                    break
        if(line == ''):
            break

#print(location_data[2][0][1])

with open("./Client_Character.csv","r") as client_data:
    data = csv.reader(client_data)
    with open("./Character_localization.csv", "w") as localized_data:
        writer = csv.writer(localized_data)
        for line in data:
            print('LINE', line)
            new_data = []
            new_data.append(line[0])
            point = line[1:]
            for i in range(512):
                if float(line[i+1]) < 0.005 and float(line[i+1]) > -0.005:
                    new_data.append(0)
                elif float(line[i+1]) > 0:
                    new_data.append(1)
                elif float(line[i+1]) < 0:
                    new_data.append(-1)
            print('NEW_DATA: ', new_data)
            writer.writerow(new_data)
"""
