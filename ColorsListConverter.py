import csv
import itertools


def convert(file_name, result_name):
    with open(result_name, "w", newline="\n") as file_writer:
        fields = ["Name", "R", "G", "B"]

        writer = csv.DictWriter(file_writer, fieldnames=fields)
        headers = {"Name": "Name", "R": "Red", "G": "Green", "B": "Blue"}
        writer.writerow(headers)

        with open(file_name, encoding='utf-8') as f:
            lis = [line.replace('\n', '').split(',') for line in f]  # create a list of lists

            for i, x in enumerate(lis):  # print the list items
                rgb = hex_to_rgb(x[0])
                output = {"Name": x[1], "R": rgb[0], "G": rgb[1], "B": rgb[2]}
                writer.writerow(output)


def hex_to_rgb(value):
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


def join_csv_files(file_names):
    filenames = file_names
    with open('result.csv', 'w') as outfile:
        for line in itertools.chain.from_iterable(map(open, filenames)):
            outfile.write(line)

    return


convert('set_1_unedited.txt', 'set_1.csv')
convert('set_2_unedited.txt', 'set_2.csv')
join_csv_files(['set_1.csv', 'set_2.csv', 'set_3.csv'])
