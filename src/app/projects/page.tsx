import { getProjects } from '@/lib/content';
import { markdownToHtml } from '@/lib/markdown';
import { pageMetadata } from '@/lib/metadata';
import { publicProjectLink } from '@/lib/site';

export const metadata = pageMetadata('Projects', 'Projects exploring machine learning through implementation and experimentation.', '/projects');

export default async function ProjectsIndex() {
  const projects = getProjects();
  return (
    <>
      <header className="page-header">
        <p className="eyebrow">From understanding to implementation</p>
        <h1>Projects</h1>
        <p>{projects.length > 0
          ? 'Things I’ve built while exploring machine learning. A place to document the problem, the approach, and what I learned along the way.'
          : 'No projects published yet. This space will collect my implementations and experiments as I document them.'}</p>
      </header>
      {await Promise.all(projects.map(async project => {
        const html = await markdownToHtml(project.content);
        return (
          <article key={project.slug} id={project.slug} className="project-detail">
            <h2>{project.title}</h2>
            <p>{project.description}</p>
            <ul className="tags" aria-label="Technologies">{project.tech.map(tech => <li key={tech}>{tech}</li>)}</ul>
            {(publicProjectLink(project.github) || publicProjectLink(project.demo)) && <div className="project-actions">
              {publicProjectLink(project.github) && <a href={project.github} className="text-link" aria-label={`Source code for ${project.title}`}>Source code <span aria-hidden="true">↗</span></a>}
              {publicProjectLink(project.demo) && <a href={project.demo} className="text-link" aria-label={`Live demo of ${project.title}`}>Live demo <span aria-hidden="true">↗</span></a>}
            </div>}
            <div className="project-body"><div className="prose" dangerouslySetInnerHTML={{ __html: html }} /></div>
          </article>
        );
      }))}
    </>
  );
}
