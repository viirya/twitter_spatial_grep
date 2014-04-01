
import gzip
import json
from common import parse_parameters
from cuda_spatial_grep import CudaSpatialGrep


def read_in_batch(gz_file, batch = 1):

    lines = []
    coordinates = []
    for line in gz_file:
        json_obj = json.loads(line)
        if 'coordinates' in json_obj and json_obj['coordinates'] != None:
            lines.append(line)
            coordinates.append(json_obj['coordinates']['coordinates'][0])
            coordinates.append(json_obj['coordinates']['coordinates'][1])
            if len(lines) >= batch:
                break

    return (lines, coordinates)
    

def prepare_bbox(bbox):

    return json.loads(bbox)    

def main():
    args = parse_parameters()
    print(args)

    cuda_obj = CudaSpatialGrep()
    bbox = prepare_bbox(args.bbox)

    gz_file = gzip.open(args.f, 'rb')
    out_file = open(args.o, 'wb')

    while 1:
        print("Loading data...")
        (lines, coordinates) = read_in_batch(gz_file, int(args.batch_n))
        if len(lines) == 0:
            break
        
        print("Spatial greping on CUDA...")
        hits = cuda_obj.cuda_spatial_grep(bbox, coordinates)
        
        idx = 0
        for hit in hits:
            if hit == 1:
                out_file.write(lines[idx])
            idx += 1    

    gz_file.close()
    out_file.close()

if __name__ == "__main__":
    main()

