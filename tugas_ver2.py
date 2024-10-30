import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import scipy.ndimage as ndi
from skimage import io, color, img_as_ubyte
from skimage import exposure
import ipywidgets as widgets
from skimage import io, img_as_float
from skimage import morphology
from pandas import DataFrame
from math import log10
import seaborn as sns
import plotly.express as px
from skimage import filters, measure
from skimage.measure import label, regionprops, regionprops_table
from skimage.transform import rotate
from skimage.draw import ellipse
import math
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
from matplotlib.colors import ListedColormap



# Fungsi untuk melakukan transformasi gambar dari RGB ke Grayscale dan inisialisasi kernel
def process_image(image):
    # Memotong gambar asli untuk menampilkan hanya sebagian
    img_cut = image[:, 0:580]
    
    # Transformasi dari RGB ke Grayscale
    img_gray = color.rgb2gray(img_cut)
    img_gray = img_as_ubyte(img_gray)  # Convert tipe data ke uint8
    
    # Inisialisasi kernel atau weights
    weights = np.full((3, 3), 1/9)
    
    return img_cut, img_gray, weights

# Inisialisasi variabel global untuk gambar
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None
if 'threshold' not in st.session_state:
    st.session_state.threshold = None


with st.sidebar:
    selected = option_menu("TUGAS 1", ["Home","Encyclopedia", "Pemrosesan dan Analisis Citra "], default_index=0)

if selected == "Home":
    st.title('Project FP Kelompok 2')
    st.subheader("Anggota kelompok")
    col1, col2 = st.columns(2)

    with col1:
        st.image("IMG_2267.jpg", caption="\nReynard Prastya Savero (5023211042)", use_column_width=True, width=150)
        st.image("IMG_2104.jpeg", caption="\nFrancisca Cindy Meilia Apsari (5023211021)", use_column_width=True, width=150)

    with col2:
        st.image("file.png", caption="\n Mavelyn Clarissa Tania (5023211004)", use_column_width=True, width=150)
        st.image("IMG_20240410_113029.jpg", caption="\n Narika Shinta (5023211057)", use_column_width=True, width=150)


if selected == "Encyclopedia":
    selected1 = st.sidebar.radio("", ["Penyakit", "Informasi"], index=0)
    
    if selected1 == 'Penyakit':
        st.markdown("<h1 style='text-align: center; color: red;'>🫀ENCYCLOPEDIA</h1>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            if st.button('Apa yang dimaksud Atopic Dermatitis?', key="button1"):
                new_title = '''<p style="font-family:Georgia; color:black; font-size: 20px; text-align: justify;">Dermatitis atopik adalah jenis eksim yang ditandai oleh peradangan kulit, sering disertai dengan kemerahan, kekeringan, dan kulit pecah-pecah. Kondisi ini bisa berlangsung dalam jangka waktu lama, bahkan bertahun-tahun. Umumnya, dermatitis atopik muncul pada area kulit seperti dahi, sekitar mata dan telinga, sisi leher, serta bagian dalam siku.</p>'''
                st.markdown(new_title, unsafe_allow_html=True)
        with col2:
            if st.button('Apa penyebab dari Atopic Dermatitis?', key="button2"):
                new_title1 = '''<p style="font-family:Georgia; color:black; font-size: 20px; text-align: Justify;">
                Penyebab atopic dermatitis meliputi beberapa faktor seperti:<br>
                1. Perubahan hormon<br>
                2. Alergi terhadap makanan, debu dan bulu hewan<br>
                3. Stres<br>
                4. Paparan udara yang dingin, kering, atau lembap
                </p>'''
                st.markdown(new_title1, unsafe_allow_html=True)

    elif selected1 == 'Informasi':
        st.markdown("<h1 style='text-align: center; color: blue;'>📖 Informasi</h1>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            if st.button('Gejala dari Atopic Dermatitis?', key="button3"):
                new_title = '''<p style="font-family:Georgia; color:black; font-size: 20px; text-align: justify;">
                1. Timbulnya ruam yang menonjol dan mengeluarkan cairan<br>
                2. Kulit yang kering dan bersisik.<br> 
                3. Kulit di telapak tangan atau area bawah mata tampak berkerut atau mengerut.<br> 
                4. Area kulit di sekitar mata terlihat lebih gelap.<br>
                5. Kulit pecah-pecah, mengelupas, hingga mengeluarkan darah.
                </p>'''
                st.markdown(new_title, unsafe_allow_html=True)
        with col2:
            if st.button('Video', key="button4"):
                content = """
                <iframe id='Video 1' width='400' height='315' src='https://www.youtube.com/embed/XE7sX_gzlS0' frameborder='0' allow='accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture' allowfullscreen></iframe>
                """
                st.markdown(content, unsafe_allow_html=True)


if selected == "Pemrosesan dan Analisis Citra ":
    selected1 = st.sidebar.radio(
        "",
        ["Open Data","Graphic Histogram","AHE & Otsu Tresholding","Morphological Filtering","Objek Labeling"],
        index=0
    )
    if selected1 == 'Open Data':

        st.markdown("<h1 style='text-align: center; color: green;'>📂 Open Data</h1>", unsafe_allow_html=True)

        # Upload gambar
        uploaded_file = st.file_uploader("Upload Gambar", type=["jpeg", "jpg", "png"])

        if uploaded_file is not None:
            # Simpan file di session_state agar bisa diakses di bagian lain
            st.session_state.uploaded_image = io.imread(uploaded_file)

            # Proses gambar: transformasi dari RGB ke grayscale
            img_cut, img_gray, weights = process_image(st.session_state.uploaded_image)

            # Menampilkan gambar asli dan grayscale berdampingan menggunakan dua kolom
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("<h3 style='text-align: center;'>Gambar Asli</h3>", unsafe_allow_html=True)
                st.image(img_cut, caption="Gambar Asli", use_column_width=True)
                st.write("Tipe gambar asli:", img_cut.dtype)
                st.write("Ukuran gambar asli:", img_cut.shape)

            with col2:
                st.markdown("<h3 style='text-align: center;'>Gambar Grayscale</h3>", unsafe_allow_html=True)
                st.image(img_gray, caption="Gambar Grayscale", use_column_width=True, clamp=True)
                st.write("Tipe gambar grayscale:", img_gray.dtype)
                st.write("Ukuran gambar grayscale:", img_gray.shape)


    elif selected1 == 'Graphic Histogram':
        st.markdown("<h1 style='text-align: center; color: blue;'>📊 Histogram</h1>", unsafe_allow_html=True)
        if st.session_state.uploaded_image is not None:
            # Menghitung histogram untuk gambar grayscale
            img_cut, img_gray, _ = process_image(st.session_state.uploaded_image)
            histogram = ndi.histogram(img_gray, min=0, max=255, bins=256)
            # Plot histogram
            fig, ax = plt.subplots()
            ax.plot(histogram)
            ax.set_xlabel('Gray Value')
            ax.set_ylabel('Number of Pixels')
            ax.set_title('Histogram of Gray Values')

            st.pyplot(fig)  


    elif selected1 == 'AHE & Otsu Tresholding':
        st.markdown("<h1 style='text-align: center; color: purple;'>🔍 Adaptive Histogram Equalization (AHE) & Otsu Thresholding</h1>", unsafe_allow_html=True)

        if st.session_state.uploaded_image is not None:
            # Proses gambar dan konversi ke grayscale
            _, img_gray, _ = process_image(st.session_state.uploaded_image)

            # Adaptive Histogram Equalization (AHE)
            img_hieq = exposure.equalize_adapthist(img_gray, clip_limit=0.9) * 255
            img_hieq = img_hieq.astype('uint8')

            # Median Filtering setelah AHE
            median_filtered = filters.median(img_hieq, np.ones((7, 7)))

            # Otsu Thresholding pada hasil Median Filtering
            threshold = filters.threshold_otsu(median_filtered)
            binary_image = median_filtered < threshold

            # Plot Gambar dalam format 2x2
            col1, col2 = st.columns(2)
            with col1:
                # Menampilkan hasil AHE
                fig, ax = plt.subplots()
                ax.imshow(img_hieq, cmap='gray')
                ax.set_title("Hasil Adaptive Histogram Equalization (AHE)")
                st.pyplot(fig)

                # Menampilkan hasil Otsu Thresholding
                fig, ax = plt.subplots()
                ax.imshow(binary_image, cmap='gray')
                ax.set_title("Hasil Otsu Thresholding pada Gambar Setelah Median Filtering")
                st.pyplot(fig)

            with col2:
                # Menampilkan hasil Median Filtering
                fig, ax = plt.subplots()
                ax.imshow(median_filtered, cmap='gray')
                ax.set_title("Hasil Median Filtering pada AHE")
                st.pyplot(fig)

                # Histogram Median Filtering
                fig, ax = plt.subplots(figsize=(5,4))
                histo_median = ndi.histogram(median_filtered, min=0, max=255, bins=256)
                ax.plot(histo_median)
                ax.set_xlabel("Gray Value")
                ax.set_ylabel("Number of Pixels")
                ax.set_title("Histogram Setelah Median Filtering")
                st.pyplot(fig)

                st.session_state.binary_image = binary_image
    elif selected1 == 'Morphological Filtering':
        st.markdown("<h1 style='text-align: center; color: teal;'>🔍 Morphological Filtering</h1>", unsafe_allow_html=True)
        
        if 'binary_image' in st.session_state:
            binary_image = st.session_state.binary_image
            
            # Remove small objects
            only_large_blobs = morphology.remove_small_objects(binary_image, min_size=100)
            
            # Fill small holes
            only_large = np.logical_not(morphology.remove_small_objects(
                np.logical_not(only_large_blobs), 
                min_size=100))
            image_segmented = only_large

            # Menyimpan hasil segmentasi untuk langkah berikutnya
            st.session_state.image_segmented = image_segmented

            # Menampilkan hasil dalam dua kolom
            col1, col2 = st.columns(2)
            
            with col1:
                # Menampilkan hasil setelah menghilangkan objek kecil
                fig, ax = plt.subplots()
                ax.imshow(only_large_blobs, cmap='gray')
                ax.set_title("Setelah Menghilangkan Objek Kecil")
                st.pyplot(fig)
            
            with col2:
                # Menampilkan hasil setelah mengisi lubang kecil
                fig, ax = plt.subplots()
                ax.imshow(image_segmented, cmap='gray')
                ax.set_title("Setelah Mengisi Lubang Kecil")
                st.pyplot(fig)

    elif selected1 == 'Objek Labeling':
        st.markdown("<h1 style='text-align: center; color: orange;'>🔍 Objek Labeling</h1>", unsafe_allow_html=True)

        if 'image_segmented' in st.session_state and 'threshold' in st.session_state:
            image_segmented = img_as_ubyte(st.session_state.image_segmented)
            
            # Membuat dua kolom untuk menampilkan gambar dalam format 2x2
            col1, col2 = st.columns(2)
            
            # Gambar pertama: Hasil segmentasi dengan kontur threshold
            with col1:
                fig, ax = plt.subplots()
                ax.imshow(image_segmented, cmap='gray')
                ax.contour(image_segmented, [st.session_state.threshold])
                ax.set_title("Segmented Image with Threshold Contour")
                st.pyplot(fig)
            
            # Gambar kedua: Labeling objek dengan warna acak
            lab_image = image_segmented
            rand_cmap = ListedColormap(np.random.rand(256, 3))
            labels, nlabels = ndi.label(lab_image)
            labels_for_display = np.where(labels > 0, labels, np.nan)
            
            with col2:
                fig, ax = plt.subplots()
                ax.imshow(lab_image, cmap='gray')
                ax.imshow(labels_for_display, cmap=rand_cmap)
                ax.axis('off')
                ax.set_title(f'Labeled Atopic Dermatitis ({nlabels})')
                st.pyplot(fig)

            # Menyaring objek yang terlalu kecil
            boxes = ndi.find_objects(labels)
            for label_ind, label_coords in enumerate(boxes):
                if label_coords is None:
                    continue
                cell = lab_image[label_coords]
                if np.product(cell.shape) < 1500: 
                    lab_image = np.where(labels == label_ind + 1, 0, lab_image)

            # Gambar ketiga: Subset dari objek yang terlabel, disusun secara 2x3
            labels, nlabels = ndi.label(lab_image)
            fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))

            # Menampilkan hingga maksimal 6 objek
            available_objects = ndi.find_objects(labels)
            for ii, obj_indices in enumerate(available_objects[:6]):  # Batas hingga maksimal 6 objek
                row, col = divmod(ii, 3)  # Menentukan baris dan kolom untuk 2x3 grid
                cell = image_segmented[obj_indices]
                axes[row, col].imshow(cell, cmap='gray')
                axes[row, col].axis('off')
                axes[row, col].set_title(f'Label #{ii+1}\nSize: {cell.shape}')
            
            # Menampilkan subplot secara horizontal di luar kolom agar tidak terpotong
            st.pyplot(fig)

            # Gambar keempat: Visualisasi orientasi objek dan bounding box
            label_img = label(lab_image)
            regions = regionprops(label_img)
            fig, ax = plt.subplots(figsize=(20, 5))
            ax.imshow(lab_image, cmap=plt.cm.gray)
            for props in regions:
                y0, x0 = props.centroid
                orientation = props.orientation
                x1 = x0 + math.cos(orientation) * 0.5 * props.minor_axis_length
                y1 = y0 - math.sin(orientation) * 0.5 * props.minor_axis_length
                x2 = x0 - math.sin(orientation) * 0.5 * props.major_axis_length
                y2 = y0 - math.cos(orientation) * 0.5 * props.major_axis_length

                ax.plot((x0, x1), (y0, y1), '-r', linewidth=2.5)
                ax.plot((x0, x2), (y0, y2), '-r', linewidth=2.5)
                ax.plot(x0, y0, '.g', markersize=15)

                minr, minc, maxr, maxc = props.bbox
                bx = (minc, maxc, maxc, minc, minc)
                by = (minr, minr, maxr, maxr, minr)
                ax.plot(bx, by, '-b', linewidth=2.5)

            st.pyplot(fig)









        


    
    



        
    






 


        
        






        





        
        
    




    


         