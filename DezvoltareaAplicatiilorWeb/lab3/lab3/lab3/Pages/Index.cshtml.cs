using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Mvc.RazorPages;

namespace lab3.Pages
{
    public class IndexModel : PageModel
    {
        private readonly ILogger<IndexModel> _logger;

        public IndexModel(ILogger<IndexModel> logger)
        {
            _logger = logger;
        }
        public Stiri[] stiri = new Stiri[]
        {
            new Stiri("Titlu", "Lead", "Autor" , new DateTime(2022, 10, 20)),

        };
        public void OnGet()
        {

        }
    }

    public class Stiri
    {
        public String titlu;
        public String lead;
        public String autor;
        public DateTime data;

        public Stiri() { }
        public Stiri(string titlu, string lead, string autor, DateTime data)
        {
            this.titlu = titlu;
            this.lead = lead;
            this.autor = autor;
            this.data = data;
        }
    }

}