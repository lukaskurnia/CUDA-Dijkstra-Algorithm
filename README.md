Spek: https://docs.google.com/document/d/1q4K14s-o0QdkauQofhhs7hX22KEdiGDxHzk80AzE2h4/edit


# OpenMPI - Parallel Dijkstra Algorithm
Dibuat oleh:
1. Lukas Kurnia Jonathan/13517006
2. Rika Dewi/13517147

## Petunjuk Penggunaan

```
$ make all node=$NODE
```

Masukkan jumlah node yang diinginkan pada parameter `$NODE`.

## Laporan Pengerjaan

### Deskripsi Solusi Paralel

Pada program yang kami buat, terdapat satu thread yang berfungsi sebagai root (master) yaitu thread dengan 
`world_rank = 0`. Setiap thread akan melakukan penelusuran pada graf dari beberapa sumber node yang disjoint antar 
thread untuk menghitung minimal distance dari sumber node tersebut terhadap node lainnya. Dengan `V` adalah jumlah 
node dalam 1 graf, dan `world_size` adalah jumlah thread yang digunakan, maka setiap thread akan melakukan penelusuran 
terhadap sebanyak `V div world_size` ditambah 1 jika `world_rank < (V mod world_size)`. Hasil dari pencarian beberapa 
sumber node ini akan disimpan dalam sebuah array bernama `sendSolution`. Dengan menggunakan fungsi `MPI_Gatherv`, 
maka array `sendSolution` akan dikirim dan dikumpulkan oleh thread root. Kemudian root akan mencetak hasil dari semua 
solusi yang terkirim dari setiap thread dan menyimpannya dalam txt.

### Analisis Solusi

Mungkin terdapat solusi lain yang menghasilkan kinerja yang lebih baik. Salah satunya dengan menerapkan 
dynamic programming untuk menyimpan minimum distance suatu node terhadap tetangganya sehingga hal ini dapat mengurangi 
waktu pencarian. 

Selain itu, dapat juga digunakan shared memory untuk menyimpan solusi yang dihasilkan daripada mengirimkan hasilnya 
ada thread root.

### Jumlah Thread

Kami menggunakan 100 buah thread. Hal ini dikarenakan pada kasus uji dengan jumlah node pada graph yang besar 
(> 100 node), ketika membandingkan 10, 50, dan 100 thread, didapatkan execution time yang lebih kecil 
dengan penggunaan 100 thread.

Untuk kasus uji dengan jumlah node yang kecil (< 100 node) penggunaan 100 buah thread cenderung menghasilkan waktu yang 
lebih lama dibanding dengan thread yang lebih sedikit. Hal ini dikarenakan banyak waktu yang digunakan untuk context 
swithcing dengan adanya thread yang banyak.

Namun, dengan mempertimbangkan banyaknya kasus uji, kami tetap memilih penggunaan 100 thread karena perbandingan waktu 
yang signifikan pada kasus uji 3000 node, yaitu  lebih dari 100 kali lipat lebih cepat dengan menggunakan 100 thread 
dibandingkan <10 thread.

### Data Pengukuran Kinerja (*)

Dengan N adalah jumlah node pada graf,
1. N = 100
    - paralel: 1600 micro-seconds
    - sequential: 12348 micro-seconds (8x lebih lambat)
2. N = 500
    - paralel: 98001 micro-seconds
    - sequential: 1416414 micro-seconds (14x lebih lambat)
3. N = 1000
    - paralel: 223769 micro-seconds
    - sequential: 12551909 micro-seconds (56x lebih lambat)
4. N = 3000
    - paralel: 3856801 micro-seconds
    - sequential: 389003817 micro-seconds (100x lebih lambat)

Output text hasil dari percobaan ini dapat dilihat pada folder `test`. Di akhir setiap output text terdapat juga 
total execution time. 

_*pengujian dilakukan pada server ITB (167.205.35.150)_

### Analisis Perbandingan Serial dan Paralel

Berdasarkan data yang terdapat pada section `Data Pengukuran Kinerja` tampak jelas bahwa program yang dijalankan secara 
paralel memiliki keunggulan dalam hal execution time dibanding program yang dijalankan secara sekuensial. Juga terlihat 
bahwa semakin banyak jumlah node, maka perbandingan waktu yang dihasilkan juga semakin signifikan. Hal ini disebabkan 
karena program paralel memanfaatkan setiap core yang ada pada processor untuk bekerja, sedangkan program sekuensial 
hanya mengandalkan sebuah core saja. Analoginya seperti sebuah bangunan yang dikerjakan oleh 100 worker dibanding dengan 
yang dikerjakan oleh hanya 1 worker saja. Perbedaannya pun tidak bersifat lieaer, namun bersifat eksponensial terhadap 
jumlah node yang ada. 
 
 
## Pembagian Tugas
- **13517006** mengerjakan fungsi: (50%)
    - writeFile, untuk melakukan penulisan file,
    - Dijkstra, mendapatkan solusi jarak terpendek untuk source tertentu,
    - minDistance, fungsi utilitas untuk dijkstra,
    - concat, untuk melakukan konkatenasi array pada C,
    - sortSolution, membuat matrix berisi solusi untuk di output ke file dari sebuah array hasil MPI_Gather,
    - Main program dengan paralel programming menggunakan openMPI.
- **13517147** mengerjakan fungsi: (50%)
    - printGraph, untuk menampilkan graph yang dibentuk dari rand(),
    - Dijkstra, mendapatkan solusi jarak terpendek untuk source tertentu,
    - minDistance, fungsi utilitas untuk dijkstra,
    - freeMatrix, free memory struktur data matrix,
    - Main program dengan paralel programming menggunakan openMPI.
