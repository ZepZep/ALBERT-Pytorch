from tokenization import FullTokenizer

tok = FullTokenizer("data/tta.vocab")

with open("data/tta.txt") as f:
    for i, line in enumerate(f):
        line = line.replace("#", "$")
        print(tok.tokenize(line))
        print(line)
        
        if i > 10:
            break
        
