


import note



def main():

    n = note.Note('data/train/txt/record-81.txt', 'data/train/con/record-81.con')
    labels = n.getTokenLabels()

    output = n.write(labels)
    with open('data/predictions/train/crf/record-81.con', 'w') as f:
        print >>f, output




if __name__ == '__main__':
    main()




