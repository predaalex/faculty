using Lab4.ContextModels;
using Lab4.Models;
using Lab4.Pages;
using Microsoft.AspNetCore.Identity;
using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Mvc.RazorPages;
using Microsoft.AspNetCore.Mvc.Rendering;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Options;

namespace Lab5.Pages
{
    public class AdaugareStireModel : PageModel
    {
        [BindProperty]
        public Stire stire { get; set; }
        public List<SelectListItem> categorii { get; set; }
        private readonly ILogger<AdaugareStireModel> _logger;
        private readonly StiriContext _stiriContext;

        public AdaugareStireModel(ILogger<AdaugareStireModel> logger, StiriContext stiriContext)
        {
            _logger = logger;
            _stiriContext = stiriContext;
        }

        public void OnGet()
        {
            stire = new Stire();

            categorii = _stiriContext.Categorie.Select(a =>
                                  new SelectListItem
                                  {
                                      Value = a.Id.ToString(),
                                      Text = a.Nume

                                  }).ToList();
        }
        public IActionResult OnPost() 
        {
            _stiriContext.Add(stire);
            _stiriContext.SaveChanges();
            return RedirectToPage("Index");
        }
    }
}
