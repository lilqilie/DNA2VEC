import numpy as np
import pandas as pds

#-------------------------------translate input sequence to encoding matrix-------------------------
def list_matrix(seqtext, seqLength):
    tensor = np.zeros((len(seqtext), seqLength, 4))
    for i in range(len(seqtext)):
        seq = seqtext[i]
        j = 0
        for s in seq:
            if s == 'A':
                tensor[i][j] = [1, 0, 0, 0]
            if s == 'T':
                tensor[i][j] = [0, 1, 0, 0]
            if s == 'C':
                tensor[i][j] = [0, 0, 1, 0]
            if s == 'G':
                tensor[i][j] = [0, 0, 0, 1]
            j += 1
    return tensor
  
def read_TIS_Fasta(filename):
    sequences = np.loadtxt(filename, dtype=np.str)
    sids = []
    seq = []
    sequence = []
    msg = ''
    sid = ''
    unknown_id = []
    for s in sequences:
        if ">" in s:
            sid = s
            sids.append(sid)
        else:
            if(len(s)<603):
                msg = 'the length of this sequence is ' + str(len(s)) +' that is less than 603(required length).'
                sequence.append(msg)
                unknown_id.append(id)
            else:
                msg = s[0:300]+s[303:603]
                sequence.append(msg)
                seq.append(msg)
    return sids, seq, sequence, unknown_id

def read_PAS_fasta(filename):
    sequences = np.loadtxt(filename, dtype=np.str)
    sids = []
    seq = []
    sequence = []
    msg = ''
    sid = ''
    unknown_id = []
    for s in sequences:
        if ">" in s:
            sid = s
            sids.append(sid)
        else:
            if (len(s) < 606):
                msg = 'the length of this sequence is ' + str(len(s)) + ' that is less than 606(required length).'
                sequence.append(msg)
                unknown_id.append(id)
            else:
                msg = s[0:300] + s[306:606]
                sequence.append(msg)
                seq.append(msg)

    return sids, seq, sequence, unknown_id



def calculate_M(organism, chr_id, ids, seq, results,site, cutoff, unknown):
    gNo = 0
    pNo = 0
    res_list1 = []
    res_list2 = []

    for i in ids:
        if i in unknown:
            Pre_value = 'Unknown'
            prepro = '-'
            if chr_id==None:
                res_list1.append([organism, i, seq[gNo], Pre_value, prepro])
            else:
                res_list2.append([organism, "chr"+chr_id, i, seq[gNo], Pre_value, prepro])

        else:
            prepro = results[pNo]
            if prepro > cutoff:
                Pre_value = 'Yes'
            else:
                Pre_value = 'No'
                prepro = 1-prepro
            prepro = "%.2f%%" % (prepro * 100)
            if chr_id==None:
                res_list1.append([organism, i, seq[gNo], Pre_value, prepro])
            else:
                res_list2.append([organism, "chr"+chr_id, i, seq[gNo], Pre_value, prepro])
            pNo += 1

        gNo += 1

    res_arr1 = np.array(res_list1)
    res_arr2 = np.array(res_list2)

    print(res_arr1)
    if ('acc' in organism) or ('don' in organism):
        df2 = pds.DataFrame(res_arr2, columns=["Signal", "Chromosome", "Sequence ID", "Sequence" ,"Is "+site+" contained?", "The Probability"])
    else:
        df2 = pds.DataFrame(res_arr1, columns=["Organism", "Sequence ID", "Sequence" ,"Is "+site+" contained?", "The probability"])
    return df2