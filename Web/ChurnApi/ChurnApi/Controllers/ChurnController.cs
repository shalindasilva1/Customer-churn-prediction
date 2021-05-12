using IronPython.Hosting;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Logging;
using Microsoft.Scripting.Hosting;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
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
        public string Predict()
        {
            var cmd = @"c:/Users/Shalinda/source/repos/shalindasilva1/ML-Project/Controllers/ModelController.py";
            return PatchParameter(cmd, "test.csv");
        }

        private string PatchParameter(string cmd, string args)
        {
            ProcessStartInfo start = new ProcessStartInfo();
            start.FileName = "C:/Users/Shalinda/AppData/Local/Programs/Python/Python39/python.exe";
            start.Arguments = string.Format("{0} {1}", cmd, args);
            start.UseShellExecute = false;
            start.RedirectStandardOutput = true;
            using (Process process = Process.Start(start))
            {
                using (StreamReader reader = process.StandardOutput)
                {
                    string result = reader.ReadToEnd();
                    Console.Write(result);
                }
            }
            return "";
        }
    }
}
