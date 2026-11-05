import type { Metadata } from 'next';
import Link from 'next/link';
import { getBooks, getPosts, getProjects } from '@/lib/content';
import { formatDate, site } from '@/lib/site';

export const metadata: Metadata = { alternates: { canonical: '/' } };

export default function Home() {
  const posts = getPosts().slice(0, 3);
  const projects = getProjects();
  const books = getBooks();
  return (
    <>
      <section className="intro" aria-labelledby="intro-title">
        <p className="eyebrow">Computer science · HCMUS</p>
        <h1 id="intro-title">Nguyen Phung<br />Nhat Khanh<span className="accent">.</span></h1>
        <p className="intro-lead">An undergraduate exploring<br className="hidden sm:block" /> intelligence through code and curiosity.</p>
        <div className="intro-details">
          <div>
            <p>I’m a second-year undergraduate at the University of Science, Vietnam National University – Ho Chi Minh City, in the Honors Program in Information Technology <span lang="vi">(Cử nhân Tài năng)</span>.</p>
            <p>I’m interested in AI/ML research. This is where I share what I build, work through the mathematics behind machine learning, and document what I’m learning.</p>
            <a href={site.github} className="text-link">Find me on GitHub <span aria-hidden="true">↗</span></a>
          </div>
          <aside className="interests" aria-labelledby="interests-title">
            <h2 id="interests-title" className="eyebrow">Academic interests</h2>
            <ul>{site.interests.map(interest => <li key={interest}>{interest}</li>)}</ul>
          </aside>
        </div>
      </section>

      {projects.length > 0 && <section className="home-section" aria-labelledby="projects-title">
        <div className="section-heading"><h2 id="projects-title">Selected projects</h2><Link href="/projects" className="text-link">View projects <span aria-hidden="true">→</span></Link></div>
        {projects.map(project => (
          <article className="featured-project" key={project.slug}>
            <div className="project-mark" aria-hidden="true">↗</div>
            <div><p className="eyebrow">Applied machine learning</p><h3><Link href={`/projects#${project.slug}`}>{project.title}</Link></h3><p>{project.description}</p><ul className="tags" aria-label="Technologies">{project.tech.map(tech => <li key={tech}>{tech}</li>)}</ul></div>
          </article>
        ))}
      </section>}

      <section className="home-section" aria-labelledby="writing-title">
        <div className="section-heading"><h2 id="writing-title">Recent writing</h2><Link href="/blog" className="text-link">All writing <span aria-hidden="true">→</span></Link></div>
        <div className="entry-list">{posts.map(post => (
          <article className="writing-entry" key={post.slug}>
            <time dateTime={post.date}>{formatDate(post.date)}</time>
            <div><h3><Link href={`/blog/${post.slug}`}>{post.title}</Link></h3><p>{post.description}</p></div>
            <span aria-hidden="true" className="entry-arrow">↗</span>
          </article>
        ))}</div>
      </section>

      <section className="home-section study-section" aria-labelledby="study-title">
        <p className="eyebrow">The learning notebook</p>
        <h2 id="study-title">Working through the fundamentals.</h2>
        <p>Derivations, intuition, and code experiments from my reading. These notes follow my learning process and grow as my understanding develops.</p>
        {books.map(book => <Link className="text-link" key={book.slug} href={`/notes/${book.slug}`}>{book.title} <span aria-hidden="true">→</span></Link>)}
      </section>
    </>
  );
}
