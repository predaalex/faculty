using Lab4.ContextModels;
using Lab4.Models;
using Lab4.Pages;
using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Mvc.RazorPages;
using Microsoft.AspNetCore.Mvc.Rendering;
using Microsoft.EntityFrameworkCore;

namespace Lab5.Pages
{
    public class EditeazaStireModel : PageModel
    {
        [BindProperty]
        public Stire stire { get; set; }
        private readonly ILogger<EditeazaStireModel> _logger;
        private readonly StiriContext _stiriContext;
        public List<SelectListItem> categorii { get; set; }
        public EditeazaStireModel(ILogger<EditeazaStireModel> logger, StiriContext stiriContext)
        {
            _logger = logger;
            _stiriContext = stiriContext;
        }
        public void OnGet(int StireId)
        {
            stire = _stiriContext.Stire.Include(stire => stire.Categorie).FirstOrDefault(x => x.Id == StireId);

            categorii = _stiriContext.Categorie.Select(a =>
                                  new SelectListItem
                                  {
                                      Value = a.Id.ToString(),
                                      Text = a.Nume

                                  }).ToList();
        }

        public IActionResult OnPost()
        {
            _stiriContext.Update(stire);
            _stiriContext.SaveChanges();
            return RedirectToPage("Index");
        }
    }
}
