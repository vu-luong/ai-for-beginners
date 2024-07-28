import gzip
import idx2numpy
import os
import csv
from PIL import Image

def save_image(image_data, image_path):
    """
    Lưu ảnh vào thư mục.

    :param image_data: Dữ liệu ảnh.
    :param image_path: Đường dẫn ảnh.
    """
    image = Image.fromarray(image_data)
    image.save(image_path)

def extract_images(file_path, output_folder):
    """
    Giải nén và lưu ảnh vào thư mục.

    :param file_path: Đường dẫn tập tin đã được nén.
    :param output_folder: Đường dẫn thư mục lưu các ảnh sau khi được giải nén.
    """

    # Tải dữ liệu ảnh từ tập tin được nén
    images_data = idx2numpy.convert_from_file(file_path)

    # Kiểm tra xem thư mục lưu ảnh đã được tạo chưa để tạo mới
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Lặp qua bộ dữ liệu ảnh và lưu từng ảnh xuống thư mục.
    for i, image_data in enumerate(images_data):
        image_file = os.path.join(output_folder, '{:05d}.png'.format(i))
        save_image(image_data, image_file)

def save_labels(file_path, output_folder, output_file):
    """
    Giải nén và lưu thông tin nhãn dữ liệu vào tập tin csv.

    :param file_path: Đường dẫn tập tin đã được nén.
    :param output_folder: Đường dẫn thư mục lưu các ảnh sau khi được giải nén.
    :param output_file: Tên tập tin csv để lưu nhãn.
    """
    # Kiểm tra xem thư mục lưu ảnh đã được tạo chưa để tạo mới.
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Đọc dữ liệu nhãn từ tập tin được nén.
    label_data = idx2numpy.convert_from_file(file_path)

    # Lưu vào tập tin csv.
    csv_file = os.path.join(
        output_folder,
        output_file
    )

    # Mở tập tin csv ở chế độ ghi.
    with open(csv_file, mode='w', newline='') as file:
        # Tạo đối tượng ghi csv.
        writer = csv.writer(file)
        
        # Ghi dữ liệu vào tập tin csv.
        for label in label_data:
            writer.writerow([label])

# Thông tin các thư mục.
source_folder = os.path.join('data', 'FashionMNIST', 'raw')
images_destination_folder = os.path.join('data', 'FashionMNIST', 'images')
labels_destination_folder = os.path.join('data', 'FashionMNIST', 'labels')

# Liệt kê danh sách các tập tin và thư mục trong thư mục data/FashionMNIST/raw.
file_list = os.listdir(source_folder)

# Lặp qua danh sách các tập tin và thư mục
for file_name in file_list:
    # Tạo đường dẫn đầy đủ cho tập tin hoặc thư mục.
    source_file = os.path.join(source_folder, file_name)
    
    # Kiểm tra có phải tập tin không.
    if os.path.isfile(source_file):
        
        # Nếu tập tin là gz thì giải nén nó
        if file_name.endswith(".gz"):
            output_file_path = os.path.join(
                source_folder,
                file_name[0:len(file_name) - len(".gz")]
            )
            with gzip.open(source_file, 'rb') as f_in:
                with open(output_file_path, 'wb') as f_out:
                    f_out.write(f_in.read())

file_list = os.listdir(source_folder)

# Lặp qua danh sách các tập tin và thư mục
for file_name in file_list:
    # Tạo đường dẫn đầy đủ cho tập tin hoặc thư mục.
    source_file = os.path.join(source_folder, file_name)
    
    # Kiểm tra có phải tập tin không.
    if os.path.isfile(source_file):
        
        # Nếu tập tin có đuôi -idx1-ubyte thì đây là tập tin chứa nhãn.
        if file_name.endswith("-idx1-ubyte"):
            # Giải nén và lưu các nhãn.
            save_labels(
                source_file,
                labels_destination_folder,
                file_name[0:len(file_name) - len("-idx1-ubyte")] + '.csv'
            )
        elif file_name.endswith("-idx3-ubyte"):
            # Giải nén và lưu các ảnh.
            destination_sub_folder = os.path.join(
                images_destination_folder,
                file_name[0:len(file_name) - len("-idx3-ubyte")]
            )
            extract_images(source_file, destination_sub_folder)
        print(f"Đã giải nén tập tin '{file_name}'")
