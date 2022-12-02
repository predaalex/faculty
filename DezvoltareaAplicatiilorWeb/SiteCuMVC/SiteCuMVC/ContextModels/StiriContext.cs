using Microsoft.EntityFrameworkCore;
using SiteCuMVC.Models;
using System.Collections.Generic;

namespace SiteCuMVC.ContextModels
{
    public class StiriContext : DbContext
    {
        public StiriContext(DbContextOptions<StiriContext> options) : base(options)
        {
        }

        public DbSet<Stire> Stire { get; set; }
        public DbSet<Categorie> Categorie { get; set; }

    }
}
