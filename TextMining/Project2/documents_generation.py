import os

# Sample Romanian content for .txt files
romanian_documents = {
    "doc11.txt": "România este o țară situată în Europa de Est, cunoscută pentru peisajele sale pitorești.",
    "doc12.txt": "Python este un limbaj de programare utilizat pe scară largă în știința datelor și învățarea automată.",
    "doc13.txt": "Munții Carpați sunt o regiune montană renumită din România, cu peisaje spectaculoase.",
    "doc14.txt": "Apache Lucene este o bibliotecă de căutare text de înaltă performanță scrisă în Java.",
    "doc15.txt": "Tika este un instrument care extrage text și metadate din diverse formate de documente.",
    "doc16.txt": "București, cunoscut și sub numele de 'Micul Paris', este cel mai mare oraș din România.",
    "doc17.txt": "Bucătăria tradițională românească include sarmale, mămăligă și cozonac.",
    "doc18.txt": "Învățarea automată permite computerelor să învețe din date și să îmbunătățească performanța.",
    "doc19.txt": "Delta Dunării este una dintre cele mai biodiverse regiuni din Europa, situată în România.",
    "doc20.txt": "Java este un limbaj de programare popular, utilizat frecvent în aplicații la nivel de întreprindere.",
    "doc21.txt": "Transilvania este o regiune istorică a României, renumită pentru castele și peisaje naturale.",
    "doc22.txt": "George Enescu este unul dintre cei mai mari compozitori români, cunoscut pentru Rapsodia Română.",
    "doc23.txt": "Castelul Bran este adesea asociat cu legenda lui Dracula și atrage mulți turiști anual.",
    "doc24.txt": "Dacia, un producător auto din România, este cunoscut pentru modelele accesibile și practice.",
    "doc25.txt": "Mihai Eminescu este considerat cel mai mare poet al literaturii române.",
    "doc26.txt": "Palatul Parlamentului din București este una dintre cele mai mari clădiri din lume.",
    "doc27.txt": "România are o istorie bogată în tradiții folclorice și obiceiuri populare.",
    "doc28.txt": "Lacul Sfânta Ana este singurul lac vulcanic din România, situat în Munții Harghita.",
    "doc29.txt": "Peștera Scărișoara adăpostește unul dintre cei mai mari ghețari subterani din Europa.",
    "doc30.txt": "Râul Olt este unul dintre cele mai lungi râuri din România, traversând țara de la nord la sud.",

}

output_folder = r"../Project1/src/main/resources"

# Generate and save the Romanian documents
for filename, content in romanian_documents.items():
    file_path = os.path.join(output_folder, filename)
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(content)

print("documents generated")