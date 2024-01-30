import tkinter as tk
from tkinter import filedialog, ttk
import pandas as pd
from preprocess import preprocessed_text
from lstm import classify_from_file, classify_text
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np


def select_file():
    file_path = filedialog.askopenfilename(
        filetypes=[("Excel files", "*.xlsx")])
    entry_file_path.delete(0, tk.END)
    entry_file_path.insert(tk.END, file_path)


def upload_data():
    file_path = entry_file_path.get()
    if not file_path:
        return

    try:
        df = pd.read_excel(file_path)
        display_data_result(df)
    except pd.errors.EmptyDataError:
        show_notification('File Excel kosong. Silakan pilih file lain.')
    except Exception as e:
        show_notification(f'Error: {str(e)}')


def preprocess_data():
    file_path = entry_file_path.get()
    if not file_path:
        return

    try:
        df = pd.read_excel(file_path)
        df['Processed_Text'] = df['Teks'].apply(preprocessed_text)
        display_preprocessed_result(df)
    except pd.errors.EmptyDataError:
        show_notification('File Excel kosong. Silakan pilih file lain.')
    except Exception as e:
        show_notification(f'Error: {str(e)}')


def classify_data():
    file_path = entry_file_path.get()
    if not file_path:
        return

    try:
        df = classify_from_file(file_path)
        display_classification_result(df)
        calculate_performance_metrics(df['label'], df['Category_Label'])
        display_confusion_matrix(df['label'], df['Category_Label'])
    except pd.errors.EmptyDataError:
        show_notification('File Excel kosong. Silakan pilih file lain.')
    except Exception as e:
        show_notification(f'Error: {str(e)}')


def classify_text_gui():
    input_text = entry_input_text.get()
    if not input_text:
        return

    predicted_label = classify_text(input_text)
    result_label.config(text=f'Hasil Klasifikasi: {predicted_label}')


def display_data_result(data_frame):
    data_result_tree.delete(*data_result_tree.get_children())
    for index, row in data_frame.iterrows():
        data_result_tree.insert('', 'end', values=(
            index + 1, row['Teks'], row['label']))


def display_preprocessed_result(data_frame):
    preprocessed_result_tree.delete(*preprocessed_result_tree.get_children())
    for index, row in data_frame.iterrows():
        preprocessed_result_tree.insert('', 'end', values=(
            index + 1, row['Processed_Text'], row['label']))


def display_classification_result(data_frame):
    classification_result_tree.delete(
        *classification_result_tree.get_children())
    for index, row in data_frame.iterrows():
        classification_result_tree.insert('', 'end', values=(
            index + 1, row['Teks'], row['Category_Label']))


def calculate_performance_metrics(y_true, y_pred):
    overall_accuracy = accuracy_score(y_true, y_pred)
    overall_precision = precision_score(y_true, y_pred, average='weighted')
    overall_recall = recall_score(y_true, y_pred, average='weighted')
    overall_f1 = f1_score(y_true, y_pred, average='weighted')

    class_labels = sorted(set(y_true))
    class_accuracy = [accuracy_score(
        y_true == label, y_pred == label) for label in class_labels]
    class_precision = precision_score(
        y_true, y_pred, labels=class_labels, average=None)
    class_recall = recall_score(
        y_true, y_pred, labels=class_labels, average=None)
    class_f1 = f1_score(y_true, y_pred, labels=class_labels, average=None)

    for item in performance_tree.get_children():
        performance_tree.delete(item)

    performance_tree.insert('', 'end', values=(
        'Overall', f'{overall_accuracy:.4f}', f'{overall_f1:.4f}', f'{overall_recall:.4f}', f'{overall_precision:.4f}'))

    for label, acc, prec, rec, f1 in zip(class_labels, class_accuracy, class_precision, class_recall, class_f1):
        label_text = {-1: 'Negative', 0: 'Neutral',
                      1: 'Positive'}.get(label, str(label))
        performance_tree.insert('', 'end', values=(
            str(label_text), f'{acc:.4f}', f'{f1:.4f}', f'{rec:.4f}', f'{prec:.4f}'))


def display_confusion_matrix(y_true, y_pred):
    conf_matrix = confusion_matrix(y_true, y_pred)

    for item in confusion_matrix_tree.get_children():
        confusion_matrix_tree.delete(item)

    tn = np.sum(conf_matrix[0, 0])
    fn1 = np.sum(conf_matrix[0, 1])
    fn2 = np.sum(conf_matrix[0, 2])
    tp = np.sum(conf_matrix[1, 1])
    fp1 = np.sum(conf_matrix[1, 0])
    fp2 = np.sum(conf_matrix[1, 2])
    tnr = np.sum(conf_matrix[2, 2])
    fnr1 = np.sum(conf_matrix[2, 0])
    fnr2 = np.sum(conf_matrix[2, 1])

    confusion_matrix_tree.insert('', 'end', values=(
        tn, fn1, fn2, tn, fn1, fn2, tnr, fnr1, fnr2))


def show_notification(message):
    notification_window = tk.Toplevel(root)
    notification_window.title('Pemberitahuan')

    notification_label = tk.Label(notification_window, text=message)
    notification_label.pack(padx=20, pady=20)

    ok_button = tk.Button(notification_window, text='OK',
                          command=notification_window.destroy)
    ok_button.pack(pady=10)


root = tk.Tk()
root.title('Facial Recognition Sentiment Analysis Long Short Term Memory')
root.configure(bg='#ADD8E6')

file_frame = tk.Frame(root, bg='#ADD8E6')
file_frame.pack()

btn_select_file = tk.Button(
    file_frame, text='Pilih File', command=select_file, bg='#87CEEB')
btn_select_file.grid(row=0, column=0)

entry_file_path = tk.Entry(file_frame, width=50)
entry_file_path.grid(row=0, column=1)

btn_upload = tk.Button(file_frame, text='Unggah',
                       command=upload_data, bg='#87CEEB')
btn_upload.grid(row=0, column=2)

btn_preprocess = tk.Button(
    file_frame, text='Preprocess', command=preprocess_data, bg='#87CEEB')
btn_preprocess.grid(row=0, column=3)

btn_classify = tk.Button(file_frame, text='Klasifikasi',
                         command=classify_data, bg='#87CEEB')
btn_classify.grid(row=0, column=4)

# Membuat PanedWindow sebagai wadah untuk result_tree, performance_tree, dan confusion_matrix_tree
paned_window = ttk.PanedWindow(root, orient='horizontal')
paned_window.pack(expand=True, fill='both', padx=10, pady=10)

# Create a Notebook widget to handle tabs
notebook = ttk.Notebook(paned_window)
notebook.pack(expand=True, fill='both', padx=10, pady=10)

# Create tabs for 'Data', 'Preprocessed', and 'Klasifikasi'
data_result_tree = ttk.Treeview(notebook, columns=(
    'No', 'Text', 'Label'), show='headings', selectmode='browse')
data_result_tree.heading('No', text='No')
data_result_tree.heading('Text', text='Text')
data_result_tree.heading('Label', text='Label')
data_result_tree.column('No', width=50, anchor='center')
data_result_tree.column('Text', width=200, anchor='w')
data_result_tree.column('Label', width=120, anchor='center')
data_result_tree.pack(expand=True, fill='both', pady=10)

preprocessed_result_tree = ttk.Treeview(notebook, columns=('No', 'Processed Text', 'Label'), show='headings',
                                        selectmode='browse')
preprocessed_result_tree.heading('No', text='No')
preprocessed_result_tree.heading('Processed Text', text='Processed Text')
preprocessed_result_tree.heading('Label', text='Label')
preprocessed_result_tree.column('No', width=50, anchor='center')
preprocessed_result_tree.column('Processed Text', width=200, anchor='w')
preprocessed_result_tree.column('Label', width=120, anchor='center')
preprocessed_result_tree.pack(expand=True, fill='both', pady=10)

classification_result_tree = ttk.Treeview(notebook, columns=('No', 'Text', 'Predicted Label'), show='headings',
                                          selectmode='browse')
classification_result_tree.heading('No', text='No')
classification_result_tree.heading('Text', text='Text')
classification_result_tree.heading('Predicted Label', text='Predicted Label')
classification_result_tree.column('No', width=50, anchor='center')
classification_result_tree.column('Text', width=200, anchor='w')
classification_result_tree.column(
    'Predicted Label', width=120, anchor='center')
classification_result_tree.pack(expand=True, fill='both', pady=10)

# Add tabs to the Notebook
notebook.add(data_result_tree, text='Data')
notebook.add(preprocessed_result_tree, text='Preprocessed')
notebook.add(classification_result_tree, text='Klasifikasi')

# Create widget for performance_tree
performance_tree = ttk.Treeview(paned_window, columns=('Data', 'Accuracy', 'F1-Score', 'Recall', 'Precision'),
                                show='headings', selectmode='browse')
performance_tree.heading('Data', text='Data')
performance_tree.heading('Accuracy', text='Accuracy')
performance_tree.heading('F1-Score', text='F1-Score')
performance_tree.heading('Recall', text='Recall')
performance_tree.heading('Precision', text='Precision')
performance_tree.column('Data', width=100, anchor='center')
performance_tree.column('Accuracy', width=100, anchor='center')
performance_tree.column('F1-Score', width=100, anchor='center')
performance_tree.column('Recall', width=100, anchor='center')
performance_tree.column('Precision', width=100, anchor='center')
performance_tree.pack(expand=True, fill='both', pady=10)

# Menambahkan widget Treeview untuk Confusion Matrix ke PanedWindow
confusion_matrix_tree = ttk.Treeview(paned_window,
                                     columns=['True Positive', 'False Positive1', 'False Positive2',
                                              'True Negative', 'False Negative1', 'False Negative2',
                                              'True Netral', 'False Netral1', 'False Netral2'],
                                     show='headings', selectmode='browse')
confusion_matrix_tree.heading('True Positive', text='TP')
confusion_matrix_tree.heading('False Positive1', text='FP1')
confusion_matrix_tree.heading('False Positive2', text='FP2')
confusion_matrix_tree.heading('True Negative', text='TN')
confusion_matrix_tree.heading('False Negative1', text='FN1')
confusion_matrix_tree.heading('False Negative2', text='FN2')
confusion_matrix_tree.heading('True Netral', text='TNR')
confusion_matrix_tree.heading('False Netral1', text='FNR1')
confusion_matrix_tree.heading('False Netral2', text='FNR2')
confusion_matrix_tree.column('True Positive', width=50, anchor='center')
confusion_matrix_tree.column('False Positive1', width=50, anchor='center')
confusion_matrix_tree.column('False Positive2', width=50, anchor='center')
confusion_matrix_tree.column('True Negative', width=50, anchor='center')
confusion_matrix_tree.column('False Negative1', width=50, anchor='center')
confusion_matrix_tree.column('False Negative2', width=50, anchor='center')
confusion_matrix_tree.column('True Netral', width=50, anchor='center')
confusion_matrix_tree.column('False Netral1', width=50, anchor='center')
confusion_matrix_tree.column('False Netral2', width=50, anchor='center')
confusion_matrix_tree.pack(expand=True, fill='both', pady=10)

# Menetapkan lebar untuk result_tree, performance_tree, dan confusion_matrix_tree
paned_window.add(notebook, weight=1)
paned_window.add(performance_tree, weight=1)
paned_window.add(confusion_matrix_tree, weight=1)

text_frame = tk.Frame(root, bg='#ADD8E6')
text_frame.pack()

label_input_text = tk.Label(text_frame, text='Masukkan Teks:', bg='#ADD8E6')
label_input_text.grid(row=0, column=0)

entry_input_text = tk.Entry(text_frame, width=50)
entry_input_text.grid(row=0, column=1)

btn_classify_text = tk.Button(
    text_frame, text='Prediksi', command=classify_text_gui, bg='#87CEEB')
btn_classify_text.grid(row=0, column=2)

result_label = tk.Label(root, text='Hasil Klasifikasi: ')
result_label.pack()

root.mainloop()
