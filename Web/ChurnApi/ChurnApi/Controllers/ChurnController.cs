using IronPython.Hosting;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Logging;
using Microsoft.Scripting.Hosting;
using System;
using System.Collections.Generic;
using System.Linq;

namespace ChurnAPI.Controllers
{
    [ApiController]
    [Route("[controller]")]
    public class ChurnController : ControllerBase
    {
        private static readonly string[] Summaries = new[]
        {
            "Freezing", "Bracing", "Chilly", "Cool", "Mild", "Warm", "Balmy", "Hot", "Sweltering", "Scorching"
        };

        private readonly ILogger<ChurnController> _logger;

        public ChurnController(ILogger<ChurnController> logger)
        {
            _logger = logger;
        }

        [HttpGet]
        public IEnumerable<WeatherForecast> Get()
        {
            var rng = new Random();
            var test = PatchParameter();
            return Enumerable.Range(1, 5).Select(index => new WeatherForecast
            {
                Date = DateTime.Now.AddDays(index),
                TemperatureC = rng.Next(-20, 55),
                Summary = Summaries[rng.Next(Summaries.Length)]
            })
            .ToArray();
        }

        private string PatchParameter()
        {
            var engine = Python.CreateEngine(); // Extract Python language engine from their grasp
            var scope = engine.CreateScope(); // Introduce Python namespace (scope)

            ScriptSource source = engine.CreateScriptSourceFromFile(@"C:\Users\Shalinda\source\repos\shalindasilva1\ML-Project\Controllers\ModelController.py"); // Load the script
            object result = source.Execute(scope);
            string parameter = scope.GetVariable<string>("parameter"); // To get the finally set variable 'parameter' from the python script
            return parameter;
        }
    }
}
