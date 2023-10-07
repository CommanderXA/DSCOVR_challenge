fin = open("./data/kp2017.txt", "r")
fout = open("./data/kp2017.csv", "w")

a = fin.readline()

value_to_remove = ""
while not a == None:
    b = a.split(" ")
    if len(b) == 0 or a == " " or a == '':
        break
    print(b)
    b = [x for x in b if x != value_to_remove]
    print(b)
    b.pop(5)
    b.pop(5)
    b.pop(-1)
    b.pop(-1)
    m = "-".join(b[0:3]) + " " + b[3].replace(".", ":") + "0:00" + ";" + b[5]
    fout.writelines([m + "\n"])
    a = fin.readline()

fin.close()
fout.close()

# print(string)
