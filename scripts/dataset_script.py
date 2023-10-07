fin = open("k_values.txt", "r")
fout = open("k_val.txt", "w")

a = fin.readline()

value_to_remove = ''
while not a == ' ' or not a == None:
    b = a.split(' ')
    print(b)
    b = [x for x in b if x != value_to_remove]
    print(b)
    b.pop(5)
    b.pop(5)
    b.pop(-1)
    b.pop(-1)
    m = '-'.join(b[0: 3]) + ' ' + b[3].replace('.', ':') + '0:00' + ";" + b[5]
    fout.writelines([m+'\n'])
    a = fin.readline()

fin.close()
fout.close()

# print(string)
