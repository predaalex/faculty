In celula cu functia extrage_tabla(img_originala) am modificati linia 63 din break in:

img_cropata = img_originala.copy()
img_cropata[:,:,:] = 0
return img_cropata

Am facut acest lucru pentru ca in unele imagini nu se gaseste tabla si nu se returneaza nimic,
asa ca returnez o imagine neagra pentru a continua rularea programului

De asemenea, setul de date a fost rulat in 170s

