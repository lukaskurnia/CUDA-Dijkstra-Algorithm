
# CUDA - Parallel Dijkstra Algorithm
Dibuat oleh:
1. Lukas Kurnia Jonathan/13517006
2. Rika Dewi/13517147

## Petunjuk Penggunaan

```
$ make node=$NODE
```

Masukkan jumlah node yang diinginkan pada parameter `$NODE`.

## Laporan Pengerjaan

### Deskripsi Solusi Paralel

Pada program yang kami buat, pada awalnya didefinisikan 3 buah variabel yang akan 
digunakan pada device dan host yaitu `graph`, `dist`, dan `sptSet`. 
* `graph` akan berisikan jarak antar node yang diinisialisasi secara random. 
* `dist` akan berisikan matriks akan menghasilkan setiap elemen a[i][j] yang merupakan 
hasil jarak terdekat dari node ke-i terhadap node ke-j yang direpresentasikan dalam 
array 1D kontigu, hal ini karena cuda hanya dapat melakukan alokasi terhadap array 1D.
* `sptSet` meruapakn matriks boolean yang menghasilkan setiap elemen a[i][j] yang berisi 
apakah node ke-i sudah pernah mendatangi node ke-j yang juga direpresentasikan dalam 
array 1D kontigu.

Dengan `V` adalah jumlah node, `blockSize` adalah jumlah thread dalam satu block, kami 
menggunakan 256 thread untuk setiap blocknya, dengan jumlah block sebanyak 
`numBlocks = (V + blockSize - 1) / blockSize`. Kemudian kedua parameter ini dipassing 
kepada fungsi global (dipanggil dari host dijalankan oleh device) bernama `cudaDjikstra`. 
Di fungsi ini kami akan memanggil `djikstra` yang memiliki type qualifier device, hal ini 
karena djikstra hanya akan dieksekusi oleh device. Fungsi `djikstra` ini akan dipanggil 
untuk setiap source node, dengan source node ini didapat dari 
`blockIdx.x * blockDim.x + threadIdx.x`. Setelah semua node sudah ditelusuri, 
kami memanggil `cudaDeviceSynchronize()` untuk melakukan sinkronisasi memory di device 
dan host. Setelah itu kami menghitung waktu yang diperlukan dan memanggil fungsi 
`writeFile` untuk `dist` yang akan menuliskan hasil djikstra ini ke file. 


### Analisis Solusi

Mungkin terdapat solusi lain yang menghasilkan kinerja yang lebih baik. Salah satunya dengan menerapkan 
dynamic programming untuk menyimpan minimum distance suatu node terhadap tetangganya sehingga hal ini dapat mengurangi 
waktu pencarian. 

Selain itu, mungkin dapat diatur parameter `blockSize` agar mendapatkan hasil yang lebih optimum untuk setiap data uji.

### Jumlah Thread

Kami menggunakan 256 buah thread. Angka ini didapat setelah melakukan research pada 
forum-forum di internet yang mengatakan lebih baik menggunakan 128/256 thread untuk setiap
block[2] (link tertera di bawah). Pemilihan 256 dibandingkan 128 adalah karena melihat 
test case kami yang cenderung besar (>256), sehingga kami memilih menggunakan 256 thread. 

### Data Pengukuran Kinerja (*)

Dengan N adalah jumlah node pada graf,
			
1. N = 100
    - paralel: 55000.66667 micro-seconds (4x lebih lambat)
    - sequential: 12348 micro-seconds 
2. N = 500
    - paralel: 1754778.667 micro-seconds (1.2x lebih lambat)
    - sequential: 1416414 micro-seconds 
3. N = 1000
    - paralel: 7801268.667 micro-seconds (1.6x lebih cepat)
    - sequential: 12551909 micro-seconds 
4. N = 3000
    - paralel: 121221839 micro-seconds (3.2x lebih cepat)
    - sequential: 389003817 micro-seconds 

Output text hasil dari percobaan ini dapat dilihat pada folder `output`. Di akhir setiap output text terdapat juga 
total execution time. 

_*pengujian dilakukan pada server ITB (167.205.32.100)_

### Analisis Perbandingan Serial dan Paralel

Berdasarkan data yang terdapat pada section `Data Pengukuran Kinerja` terlihat untuk test case kecil program sequential
lebih unggul dalam hal execution time dibanding dengan program paralel. Hal ini karena terdapat waktu untuk menyiapkan
thread dan block pada GPU, dan menyiapkan memori pada device dan host. Selain itu dibutuhkan waktu untuk sinkronisasi 
kedua memori ini sehingga jika dibandingkan dengan program sequential, maka hal ini lebih membutuhkan waktu komputasi 
yang lebih lama dibandingkan menghitung program djikstranya sendiri. Hal ini terlihat jelas apalagi ketika program
kami dijalankan dengan menggunakan command `nvprof`, terlihat bahwa waktu eksekusi fungsi global kami saja sebenarnya 
sangatlah cepat (dalam hitungan micro sedond saja).


Sedangkan, untuk test case yang besar paralel memiliki keunggulan dalam hal execution time dibanding program yang 
dijalankan secara sekuensial. Juga terlihat 
bahwa semakin banyak jumlah node, maka perbandingan waktu yang dihasilkan juga semakin signifikan. Hal ini disebabkan 
karena program paralel memanfaatkan lebih banyak worker untuk bekerja, sedangkan program sekuensial 
hanya mengandalkan sebuah core saja. Analoginya seperti sebuah bangunan yang dikerjakan oleh 100 worker dibanding dengan 
yang dikerjakan oleh hanya 1 worker saja. Perbedaannya pun tidak bersifat linear, namun bersifat eksponensial terhadap 
jumlah node yang ada. 
 
## Pembagian Tugas
- **13517006** (50%)
- **13517147** (50%)

Selama mengerjakan tugas ini, kami tidak secara eksplisit membagi fungsi yang akan dikerjakan, namun kami berdiskusi 
untuk setiap langkah yang kami ambil dan melakukan coding secara bersamaan sembari berdiskusi. Oleh karena itu,
dapat dikatakan kami tidak membagi tugas yang ada melainkan mengerjakannya secara bersamaan. 

## Reference

[1] Specification
> https://docs.google.com/document/d/1q4K14s-o0QdkauQofhhs7hX22KEdiGDxHzk80AzE2h4/edit

[2] Optimal number of thread in CUDA programming
> https://www.researchgate.net/post/The_optimal_number_of_threads_per_block_in_CUDA_programming