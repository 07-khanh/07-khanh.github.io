import { getProjects } from '@/lib/content';
import { markdownToHtml } from '@/lib/markdown';

export default async function ProjectsIndex() {
  const projects = getProjects();
  
  return (
    <div className="space-y-12">
      <header className="border-b border-gray-200 pb-8">
        <h1 className="text-3xl font-sans font-bold text-gray-900">Projects</h1>
        <p className="text-gray-600 mt-2 font-serif">A selection of things I've built.</p>
      </header>
      
      <div className="space-y-12">
        {await Promise.all(projects.map(async (project) => {
          const html = await markdownToHtml(project!.content);
          return (
            <section key={project!.slug} className="bg-white p-8 rounded-2xl border border-gray-200 shadow-sm">
              <div className="flex flex-col md:flex-row justify-between items-start gap-4 mb-6">
                <div>
                  <h2 className="text-2xl font-sans font-bold text-gray-900">{project!.title}</h2>
                  <p className="text-gray-500 font-serif mt-2 leading-relaxed">{project!.description}</p>
                </div>
                <div className="flex gap-3 font-sans text-sm shrink-0">
                  {project!.github && (
                    <a href={project!.github} target="_blank" rel="noopener noreferrer" className="px-4 py-2 bg-gray-100 text-gray-700 hover:bg-gray-200 rounded-lg font-semibold transition-colors">GitHub</a>
                  )}
                  {project!.demo && project!.demo !== "#" && (
                    <a href={project!.demo} target="_blank" rel="noopener noreferrer" className="px-4 py-2 bg-gray-900 text-white hover:bg-gray-800 rounded-lg font-semibold transition-colors">Live Demo</a>
                  )}
                </div>
              </div>
              
              <div className="flex flex-wrap gap-2 mb-6">
                {project!.tech.map((t: string) => (
                  <span key={t} className="px-2 py-1 bg-amber-50 text-amber-700 text-xs font-mono font-medium rounded border border-amber-200/50">
                    {t}
                  </span>
                ))}
              </div>
              
              <div className="prose prose-sm max-w-none font-serif text-gray-700 border-t border-gray-100 pt-6" dangerouslySetInnerHTML={{ __html: html }} />
            </section>
          )
        }))}
      </div>
    </div>
  );
}
