using Lab4.ContextModels;
using Lab4.Models;
using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Mvc.RazorPages;
using Microsoft.AspNetCore.Mvc.Rendering;
using Microsoft.EntityFrameworkCore;

namespace Lab5.Pages
{
    public class StergeStireModel : PageModel
    {
        public static Stire stire { get; set; }
        private readonly ILogger<StergeStireModel> _logger;
        private readonly StiriContext _stiriContext;
        public StergeStireModel(ILogger<StergeStireModel> logger, StiriContext stiriContext)
        {
            _logger = logger;
            _stiriContext = stiriContext;
        }
        public void OnGet(int StireId)
        {
            stire = _stiriContext.Stire.Include(stire => stire.Categorie).FirstOrDefault(stire => stire.Id == StireId);
            
        }

        public IActionResult OnPost()
        {
            if (stire != null)
            {
                _stiriContext.Remove(stire);
                _stiriContext.SaveChanges();
            }
            return RedirectToPage("Index");
        }
    }
}
