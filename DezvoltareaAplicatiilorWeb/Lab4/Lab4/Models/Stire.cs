﻿namespace Lab4.Models
{
    public class Stire
    {
        public int Id { get; set; }
        public string Titlu { get; set; }

        public string Lead { get; set; }
        public string Continut { get; set; }
        public string Autor { get; set; }
        public int IdCategorie { get; set; }
        public Categorie Categorie { get; set; }
    }

}
