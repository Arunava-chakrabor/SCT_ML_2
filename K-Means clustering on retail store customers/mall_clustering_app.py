import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from tkinter import *
from tkinter import filedialog, messagebox, ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

df = None

# Load dataset
def load_data():
    global df
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if file_path:
        df = pd.read_csv(file_path)
        preview_data(df, "Original Dataset")
        messagebox.showinfo("Success", "Dataset loaded successfully!")

# Preview dataset in Treeview
def preview_data(data, title):
    for widget in dataset_frame.winfo_children():
        widget.destroy()

    Label(dataset_frame, text=title, font=("Arial", 14, "bold")).pack(pady=5)

    tree_frame = Frame(dataset_frame)
    tree_frame.pack(fill=BOTH, expand=True)

    tree_scroll = Scrollbar(tree_frame)
    tree_scroll.pack(side=RIGHT, fill=Y)

    tree = ttk.Treeview(tree_frame, yscrollcommand=tree_scroll.set)
    tree.pack(fill=BOTH, expand=True)
    tree_scroll.config(command=tree.yview)

    tree["columns"] = list(data.columns)
    tree["show"] = "headings"

    for col in data.columns:
        tree.heading(col, text=col)
        tree.column(col, width=100, anchor=CENTER)

    for _, row in data.iterrows():
        tree.insert("", "end", values=list(row))

# Run K-Means clustering
def run_clustering():
    global df
    if df is None:
        messagebox.showerror("Error", "Please load the dataset first!")
        return

    try:
        n_clusters = int(cluster_spinbox.get())
        X = df.iloc[:, [3, 4]].values  # Annual Income & Spending Score

        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        df['Cluster'] = kmeans.fit_predict(X)

        fig, ax = plt.subplots(figsize=(6, 4))
        colors = plt.cm.tab10.colors
        for i in range(n_clusters):
            ax.scatter(X[df['Cluster'] == i, 0], X[df['Cluster'] == i, 1],
                       s=50, c=[colors[i % 10]], label=f'Cluster {i+1} ({(df["Cluster"]==i).sum()} customers)')
        ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                   s=200, c='yellow', label='Centroids', edgecolors='black')
        ax.set_title(f'Customer Clusters (k={n_clusters})', fontsize=14)
        ax.set_xlabel('Annual Income (k$)')
        ax.set_ylabel('Spending Score (1-100)')
        ax.legend()

        # Show plot in Tkinter
        for widget in plot_frame.winfo_children():
            widget.destroy()
        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()

        preview_data(df, "Dataset with Clusters")

    except Exception as e:
        messagebox.showerror("Error", str(e))

# Tkinter GUI setup
window = Tk()
window.title("Mall Customer Segmentation")
window.geometry("1000x700")
window.configure(bg="#f4f4f4")

title_label = Label(window, text="🛍 Mall Customer Clustering", font=("Arial", 18, "bold"), bg="#f4f4f4")
title_label.pack(pady=10)

# Buttons Frame
button_frame = Frame(window, bg="#f4f4f4")
button_frame.pack(pady=5)

Button(button_frame, text="📂 Load Dataset", command=load_data, width=20, bg="#4CAF50", fg="white",
       font=("Arial", 12, "bold")).grid(row=0, column=0, padx=10)

Label(button_frame, text="Clusters:", font=("Arial", 12), bg="#f4f4f4").grid(row=0, column=1, padx=5)
cluster_spinbox = Spinbox(button_frame, from_=2, to=10, width=5, font=("Arial", 12))
cluster_spinbox.grid(row=0, column=2, padx=5)

Button(button_frame, text="⚡ Run Clustering", command=run_clustering, width=20, bg="#2196F3", fg="white",
       font=("Arial", 12, "bold")).grid(row=0, column=3, padx=10)

# Frames for Dataset & Plot
dataset_frame = Frame(window, bg="white", relief=SOLID, bd=1)
dataset_frame.pack(fill=BOTH, expand=True, padx=10, pady=5)

plot_frame = Frame(window, bg="white", relief=SOLID, bd=1)
plot_frame.pack(fill=BOTH, expand=True, padx=10, pady=5)

window.mainloop()
