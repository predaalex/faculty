using Lab4.ContextMdels;
using Lab4.Models;
using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Mvc.RazorPages;
using Microsoft.EntityFrameworkCore;

namespace Lab4.Pages
{
    public class StireModel : PageModel
    {
        private readonly ILogger<IndexModel> _logger;
        private readonly StiriContext _stiriContext;
        public StireModel(ILogger<IndexModel> logger, StiriContext stiriContext)
        {
            _stiriContext = stiriContext;
            _logger = logger;
        }
        public Stire Stire { get; set; }
    
        public IActionResult OnGet(int stireId)
        {
            Stire = _stiriContext.Stire.Include(Stire => Stire.Categorie).FirstOrDefault(Stire => Stire.Id == stireId);
            if(Stire == null)
            {
                return RedirectToPage("Error");
            }
            return Page();
        }
    }
}
