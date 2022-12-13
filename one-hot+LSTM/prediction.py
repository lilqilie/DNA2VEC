import argparse
import sys
import os

from Model import model
from util import list_matrix, read_TIS_Fasta, read_PAS_fasta, calculate_M

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)



#seqLength = 600  # TIS and polyA is 600, splice is 398
#gsr = 'TIS' # TIS/ PAS / PAS_AATAAA / PAS_miniData
#organism = 'hs' # hs / mm /  bt / dm


#------------------------- Arguments Parser ------------------------------------------
parser = argparse.ArgumentParser(description='Parameters to be used for DeepGenGrep training.')
parser.add_argument('-gsr', '--GSR', help='Genome signal region: TIS, PAS,PAS_AATAAA, Splice_acc, Splice_don', required=True)
parser.add_argument('-org', '--Organism', help='Organism name', required=True)
parser.add_argument('-input','--inputDataFileName',help='input file name',required = True) 
# the length sequence in the input file is 600 for TIS, PAS, and 398 for splice site
parser.add_argument('-out', '--outdir', required=True, help='the output directory')


args = parser.parse_args()

gsr = args.GSR
organism = args.Organism
intputFileName=args.inputDataFileName
out_dir=args.outdir


#-----------------------print input arguments------------------------------------------------
print('gsr:',gsr)
print('organism:',organism)
print('input file:',intputFileName)
print('the output directory:',out_dir)

results_path = out_dir + '/prediction_result.csv'
weight_file = 'Model/' + gsr + '/' + organism + '_bestModel.hdf5'

GSR_model =  model(600)
pred_list = None
if gsr == 'TIS':    
    sid, seq, sequence, unknownID = read_TIS_Fasta(intputFileName)
    seq = list_matrix(seq, 600)
    GSR_model.load_weights(weight_file)
    pred_list = GSR_model.predict(seq)
    
if gsr == 'PAS':
    sid, seq, sequence, unknownID = read_PAS_fasta(intputFileName)
    seq = list_matrix(seq, 600)
    GSR_model.load_weights(weight_file)
    pred_list = GSR_model.predict(seq)

print(pred_list)
res = calculate_M(organism, chr_id=None, ids=sid, seq=sequence, results=pred_list, site=gsr,cutoff=0.5, unknown=unknownID)
res.to_csv(results_path)
