using Lab4.ContextMdels;
using Lab4.Models;
using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Mvc.RazorPages;

namespace Lab4.Pages
{
    public class AdaugareStireModel : PageModel

    {
        private readonly ILogger<AdaugareStireModel> _logger;
        private readonly StiriContext _stiriContext;

        public AdaugareStireModel(ILogger<AdaugareStireModel> logger, StiriContext stiriContext)
        {
            _stiriContext = stiriContext;
            _logger = logger;
        }

        [BindProperty]
        public Stire Stire { get; set; }

        public void OnGet()
        {
            Stire = new Stire();
        }
        public void OnPost()
        {
            _stiriContext.Add(Stire);
            _stiriContext.SaveChanges();
        }
    }
}
