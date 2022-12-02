using Lab4.Models;
using Microsoft.AspNetCore.Identity.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore;

namespace Lab4.ContextModels
{
    public class StiriContext : IdentityDbContext
    {
        public StiriContext (DbContextOptions<StiriContext> options) : base(options)
        { 
        }

        public DbSet<Stire> Stire { get; set; }
        public DbSet<Categorie> Categorie { get; set; }

    }
}
