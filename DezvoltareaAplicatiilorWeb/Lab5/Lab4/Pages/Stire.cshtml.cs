using Lab4.ContextModels;
//using Lab4.Migrations;
using Lab4.Models;
using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Mvc.RazorPages;
using Microsoft.EntityFrameworkCore;

namespace Lab4.Pages
{
    public class StireModel : PageModel
    {
        public Stire Stire { get; set; }
        private readonly ILogger<StireModel> _logger;
        private readonly StiriContext _stiriContext;

        public StireModel(ILogger<StireModel> logger, StiriContext stiriContext)
        {
            _logger = logger;
            _stiriContext = stiriContext;
        }
        public void OnGet(int StireId)
        {
            Stire = _stiriContext.Stire.Include(stire => stire.Categorie).FirstOrDefault(x => x.Id == StireId);
        }
    }
}
