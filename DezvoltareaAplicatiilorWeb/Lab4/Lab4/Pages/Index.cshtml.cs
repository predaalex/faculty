using Lab4.ContextMdels;
using Lab4.Models;
using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Mvc.RazorPages;
using Microsoft.EntityFrameworkCore;

namespace Lab4.Pages
{
    public class IndexModel : PageModel
    {
        private readonly ILogger<IndexModel> _logger;
        private readonly StiriContext _stiriContext;
        public Stire[] Stiri;

        public IndexModel(ILogger<IndexModel> logger, StiriContext stiriContext)
        {
            _stiriContext = stiriContext;
            _logger = logger;
        }

        public void OnGet()
        {
            Stiri = _stiriContext.Stire.Include(Stire => Stire.Categorie).ToArray();
        }
    }
}