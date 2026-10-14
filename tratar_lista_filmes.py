f1 = open("movies.csv", "r", -1, "utf-8")
idxs_comma: list[int] = []
for l in f1:
    idx_comma = len(l)
    for char in l[::-1]:
        idx_comma -= 1
        if char == ",":
            idxs_comma.append(idx_comma)
            break

idxs_comma = idxs_comma[1:]

print("Etapa 1")

i = -1
result = list()
f1.close()

f2 = open("movies.csv", "r", -1, "utf-8")
f2.readline()
for l in f2:
    i += 1
    nome_filme = l[l.find(",")+1:idxs_comma[i]]
    categorias = l[idxs_comma[i]+1:]
    categorias = categorias.replace("\n", "")
    # categorias = categorias.split("|")
    result.append((nome_filme, categorias))

print("Etapa 2")

str_to_write = ""
for l in result:
    str_to_write += f"{l[0].replace('"', '')};{l[1]}\n"

print("Etapa 3")

f3 = open("lista_filmes_tratada_maior.csv", "wt", -1, "utf-8")
f3.write(str(str_to_write))