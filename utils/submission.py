from tqdm import tqdm as tqdm

def save_results_as_csv(list_results, out_file):
    f= open(out_file,"w+")
    results="ImageId,PredictionString\n"
    f.write(results)
    for id, arr_pred in tqdm(list_results):
        results=id+","
        for pred in arr_pred:
            results+= str(pred[0]) + " "+ str(pred[1])+ " " + ' '.join(str(e) for e in pred[2])+ " "
        results +="\n"
        f.write(results)
    f.close() 