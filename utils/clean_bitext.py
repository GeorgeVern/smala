import argparse


def main(args):
    with open(args.bitxt) as file_in:
        with open(args.save, "w+") as file_out:
            for line in file_in:
                biline = line.rstrip().split('|||')
                if len(biline[0]) > 1 and len(biline[1]) > 1:
                    file_out.write(line)

if __name__ == "__main__":
    parser = argparse.ArgumentParser('clean parallel corpus')
    parser.add_argument('--bitxt', help='where the bitext is stored')
    parser.add_argument('--save', help='where the output data will be saved')
    args = parser.parse_args()
    main(args)