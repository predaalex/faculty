using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Mvc.Rendering;
using Microsoft.EntityFrameworkCore;
using SiteCuMVC.ContextModels;
using SiteCuMVC.Models;
using System.Drawing;

namespace SiteCuMVC.Controllers
{
    public class StiriController : Controller
    {
        StiriContext _stiriContext { get; set; }
        public  StiriController (StiriContext stiriContext)
        {
            _stiriContext = stiriContext;
        }

        public IActionResult Index()
        {
            var stiri = _stiriContext.Stire.Include(stire => stire.Categorie).ToArray();
            return View("Stiri", stiri);
        }

        public IActionResult ShowAdaugareStire()
        {
            ViewBag.categorii = _stiriContext.Categorie.Select(a =>
                                  new SelectListItem
                                  {
                                      Value = a.Id.ToString(),
                                      Text = a.Nume

                                  }).ToList();
            return View("AdaugareStire");
        }
        [HttpPost]
        public IActionResult AdaugareStire(Stire stire)
        {
            _stiriContext.Add(stire);
            _stiriContext.SaveChanges();
            return RedirectToAction("Index");
        }

        public IActionResult ShowEditeazaStire(int Id)
        {
            ViewBag.categorii = _stiriContext.Categorie.Select(a =>
                                  new SelectListItem
                                  {
                                      Value = a.Id.ToString(),
                                      Text = a.Nume

                                  }).ToList();
            var stire = _stiriContext.Stire.Include(stire => stire.Categorie).FirstOrDefault(x => x.Id == Id);
            return View("EditeazaStire", stire);
        }

        public IActionResult EditeazaStire(Stire stire)
        {
            _stiriContext.Update(stire);
            _stiriContext.SaveChanges();
            return RedirectToAction("Index");
        }

        public IActionResult Delete(int Id)
        {
            var stire = _stiriContext.Stire.Include(stire => stire.Categorie).FirstOrDefault(x => x.Id == Id);
            _stiriContext.Remove(stire);
            _stiriContext.SaveChanges();
            return RedirectToAction("Index");
        }
    }
}
