using System.Collections.Generic;
using BERT.WebApi.ViewModels;
using Microsoft.AspNetCore.Mvc;
using Microsoft.ML.Models.BERT;

namespace BERT.WebApi.Controllers
{
    [Route("api")]
    [ApiController]
    public class PredictionController : ControllerBase
    {
        private readonly BertModel _bertModel;

        public PredictionController(BertModel bertModel)
        {
            _bertModel = bertModel;
        }

        [HttpPost]
        [Route("predict")]
        public ActionResult<IEnumerable<string>> Predict(QuestionViewModel question)
        {
            var (tokens, probability) = _bertModel.Predict(question.Context, question.Question);

            return Ok(new PredictionViewModel() {
                Tokens = tokens,
                Probability = probability
            });
        }
    }
}
