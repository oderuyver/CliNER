import os
import os.path
import sys
import argparse
import glob
import tools

from documents import *

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("-t",
        help = "Text files that were used to generate predictions",
        dest = "txt",
    )

    parser.add_argument("-c",
        help = "The directory that contains predicted concept files organized into subdirectories for svm, lin, srf",
        dest = "con",
    )

    parser.add_argument("-r",
        help = "The directory that contains reference gold standard concept files",
        dest = "ref",
    )

    parser.add_argument("-f",
        dest = "format",
        help = "Data format (i2b2 or xml).",
        default = 'i2b2'
    )

    parser.add_argument("--concept",
        dest = "do_concept",
        help = "A flag indicating whether to evaluate chunk-level or concept-level",
        action = "store_true",
        default = False
    )


    parser.add_argument("-o",
        help = "Write the evaluation to a file rather than STDOUT",
        dest = "output",
        default = None
    )

    # Parse command line arguments
    args = parser.parse_args()


    # Is output destination specified?
    if args.output:
        args.output = open(args.output, "w")
    else:
        args.output = sys.stdout


    # Which format to read?
    if   args.format == 'i2b2':
        wildcard = '*.con'
    elif args.format == 'xml':
        wildcard = '*.xml'
    else:
        print >>sys.stderr, '\n\tError: Must specify output format (i2b2 or xml)'
        print >>sys.stderr, ''
        exit(1)


    # List of medical text
    txt_files = glob.glob(args.txt)
    txt_files_map = tools.map_files(txt_files)


    # List of gold data
    ref_files = glob.glob( os.path.join(args.ref, wildcard) )
    ref_files_map = tools.map_files(ref_files)


    # List of predictions
    pred_files = glob.glob( os.path.join(args.con, wildcard) )
    pred_files_map = tools.map_files(pred_files)


    # Grouping of text, predictions, gold
    files = []
    for k in txt_files_map:
        if k in pred_files_map and k in ref_files_map:
            files.append((txt_files_map[k], pred_files_map[k], ref_files_map[k]))

    if args.do_concept:
        tag2id = { 'problem':0, 'test':1, 'treatment':2, 'none':3 }
    else:
        from documents import labels as tag2id


    # Compute the confusion matrix
    confusion = [[0] * len(tag2id) for e in tag2id]


    # txt          <- medical text
    # annotations  <- predictions
    # gold         <- gold standard
    for txt, annotations, gold in files:

        # Read predictions and gols standard data
        cnote = Document(txt, annotations)
        rnote = Document(txt, gold)

        '''
        # List of list of labels
        predictions = tools.flatten( cnote.conlist() )
        gold        = tools.flatten( rnote.conlist() )

        for p,g in zip(predictions,gold):
            if args.do_concept:
                p = p[2:]
                g = g[2:]
                if p == '': p = 'none'
                if g == '': g = 'none'
            confusion[tag2id[g]][tag2id[p]] += 1
        '''

        #'''
        sents       = cnote.getTokenizedSentences()
        predictions = cnote.conlist()
        gold        = rnote.conlist()
        for i,(pline,gline) in enumerate(zip(predictions,gold)):
            #for p,g in zip(pline,gline)[1:]:
            #for p,g in zip(pline,gline)[:1]:
            for j,(p,g) in enumerate(zip(pline,gline)):
                # try to ignore those leading articles
                #if j < len(pline)-1:
                    #if pline[j+1][2:]==gline[j+1][2:] and pline[j+1][0]=='B' and gline[j+1][0]=='I':
                    #if pline[j+1][2:]==gline[j+1][2:] and p=='B' and gline[i+1][0]=='B':
                    #    continue

                #if sents[i][j] == '__num__':
                #    continue

                #if j == 0:
                #    continue

                if args.do_concept:
                    p = p[2:]
                    g = g[2:]
                    if p == '': p = 'none'
                    if g == '': g = 'none'
                confusion[tag2id[g]][tag2id[p]] += 1
        #'''




    # Display the confusion matrix
    if args.do_concept:
        choice = 'CONCEPT'
    else:
        choice = '7-way'
    print >>args.output, ""
    print >>args.output, ""
    print >>args.output, ""
    print >>args.output, "================"
    print >>args.output, "%s RESULTS" % choice
    print >>args.output, "================"
    print >>args.output, ""
    print >>args.output, "Confusion Matrix"
    pad = max(len(l) for l in tag2id) + 6
    print >>args.output, "%s %s" % (' ' * pad, "\t".join([s[:5] for s  in tag2id.keys()]))
    for act, act_v in tag2id.items():
        print >>args.output, "%s %s" % (act.rjust(pad), "\t".join([str(confusion[act_v][pre_v]) for pre, pre_v in tag2id.items()]))
    print >>args.output, ""

    # Compute the analysis stuff
    precision = []
    recall = []
    specificity = []
    f1 = []

    tp = 0
    fp = 0
    fn = 0
    tn = 0

    print >>args.output, "Analysis"
    print >>args.output, " " * pad, "%10s%10s%10s" % ("Precision","Recall","F1")

    for lab, lab_v in tag2id.items():
        tp = confusion[lab_v][lab_v]
        fp = sum(confusion[v][lab_v] for k, v in tag2id.items() if v != lab_v)
        fn = sum(confusion[lab_v][v] for k, v in tag2id.items() if v != lab_v)
        tn = sum(confusion[v1][v2] for k1, v1 in tag2id.items()
          for k2, v2 in tag2id.items() if v1 != lab_v and v2 != lab_v)
        precision += [float(tp) / (tp + fp + 1e-100)]
        recall += [float(tp) / (tp + fn + 1e-100)]
        specificity += [float(tn) / (tn + fp + 1e-100)]
        f1 += [float(2 * tp) / (2 * tp + fp + fn + 1e-100)]
        print >>args.output, "%s %10.4f%10.4f%10.4f" % (lab.rjust(pad), precision[-1], recall[-1], f1[-1])

    print >>args.output, "--------"

    precision = sum(precision) / len(precision)
    recall = sum(recall) / len(recall)
    specificity = sum(specificity) / len(specificity)
    f1 = sum(f1) / len(f1)

    print >>args.output, "Average: %.4f\t%.4f\t%.4f" % (precision, recall, f1)


if __name__ == '__main__':
    main()
