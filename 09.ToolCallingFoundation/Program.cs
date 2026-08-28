using System.ComponentModel;
using Google.GenAI;
using Microsoft.Extensions.AI;

var builder=WebApplication.CreateBuilder(args);
var projectId=builder.Configuration["GoogleCloud:ProjectId"]??throw new InvalidOperationException("Set GoogleCloud:ProjectId.");
var location=builder.Configuration["GoogleCloud:Location"]??"global";
var model=builder.Configuration["GoogleCloud:GeminiModel"]??"gemini-2.5-flash";
var genai=new Client(project:projectId,location:location,enterprise:true);

IChatClient chatClient=genai.AsIChatClient(model).AsBuilder().UseFunctionInvocation().Build();

var packageService=new DemoPackageService();

var app=builder.Build();

app.MapPost("/agent",async(AgentRequest request)=>
{
    var tools=new AITool[]
    {
        AIFunctionFactory.Create(
            method: ([Description("Package tracking number")] string trackingNumber)=>packageService.GetStatus(trackingNumber),
            name:"get_package_status",
            description:"Gets current package status and hours since the last tracking movement."),
        AIFunctionFactory.Create(
            method: ([Description("Search phrase for company policy")] string query)=>packageService.SearchPolicy(query),
            name:"search_company_policy",
            description:"Searches the authoritative company policy knowledge base for package rules."),
        AIFunctionFactory.Create(
            method: ([Description("Package tracking number")] string trackingNumber,[Description("Reason for investigation")] string reason)=>packageService.CreateInvestigationCase(trackingNumber,reason),
            name:"create_investigation_case",
            description:"Creates an investigation case. Use only when retrieved policy and package state indicate an investigation is required.")
    };

    var response=await chatClient.GetResponseAsync(request.Message,new ChatOptions{Tools=tools});
    return Results.Ok(new{answer=response.Text});
});
app.Run();

public sealed record AgentRequest(string Message);
public sealed class DemoPackageService
{
    public object GetStatus(string trackingNumber) => new{trackingNumber,status="InTransit",lastMovementHours=53};
    public object SearchPolicy(string query) => new{query,policy="Packages with no tracking movement for more than 48 hours require an investigation case."};
    public object CreateInvestigationCase(string trackingNumber,string reason) => new{caseId=$"INV-{Random.Shared.Next(10000,99999)}",trackingNumber,reason};
}
