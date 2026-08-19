import Link from 'next/link';
import { getAllBlogPosts, getAllProjects, getAllBooks } from '@/lib/content';
import { BookOpen, FolderGit2, FileText, ArrowRight } from 'lucide-react';

export default function HomePage() {
  const recentPosts = getAllBlogPosts().slice(0, 3);
  const featuredProjects = getAllProjects().filter((p) => p.featured).slice(0, 2);
  const books = getAllBooks();

  return (
    <div className="space-y-16">
      <section className="space-y-4">
        <h1 className="text-4xl font-bold tracking-tight">Technical Knowledge Base & Portfolio</h1>
        <p className="text-xl text-neutral-600 dark:text-neutral-400 max-w-2xl">
          Hi, I&apos;m building scalable systems and studying machine learning. This is my digital garden, study notebook, and open workspace.
        </p>
        <div className="flex gap-4 pt-2">
          <Link href="/notes" className="flex items-center gap-2 px-4 py-2 bg-neutral-900 text-white rounded-md text-sm hover:bg-neutral-800">
            <BookOpen size={16} /> Read Study Notes
          </Link>
          <Link href="/projects" className="flex items-center gap-2 px-4 py-2 border rounded-md text-sm hover:bg-neutral-100 dark:hover:bg-neutral-800">
            <FolderGit2 size={16} /> View Projects
          </Link>
        </div>
      </section>

      {/* RECENT POSTS */}
      <section className="space-y-6">
        <div className="flex justify-between items-center">
          <h2 className="text-2xl font-semibold flex items-center gap-2">
            <FileText size={20} /> Latest Articles
          </h2>
          <Link href="/blog" className="text-sm font-medium flex items-center gap-1 hover:underline">
            All posts <ArrowRight size={14} />
          </Link>
        </div>
        <div className="grid gap-4">
          {recentPosts.map((post) => (
            <Link key={post.slug} href={`/blog/${post.slug}`} className="p-4 rounded-lg border hover:border-neutral-400 transition-colors">
              <div className="flex justify-between items-baseline mb-1">
                <h3 className="text-lg font-medium">{post.title}</h3>
                <span className="text-xs text-neutral-500 font-mono">{post.date}</span>
              </div>
              <p className="text-sm text-neutral-600 dark:text-neutral-400">{post.description}</p>
            </Link>
          ))}
        </div>
      </section>

      {/* FEATURED PROJECTS */}
      <section className="space-y-6">
        <div className="flex justify-between items-center">
          <h2 className="text-2xl font-semibold flex items-center gap-2">
            <FolderGit2 size={20} /> Featured Projects
          </h2>
          <Link href="/projects" className="text-sm font-medium flex items-center gap-1 hover:underline">
            All projects <ArrowRight size={14} />
          </Link>
        </div>
        <div className="grid md:grid-cols-2 gap-4">
          {featuredProjects.map((project) => (
            <Link key={project.slug} href={`/projects/${project.slug}`} className="p-5 border rounded-lg hover:border-neutral-400 transition-colors flex flex-col justify-between">
              <div>
                <h3 className="text-lg font-medium mb-2">{project.title}</h3>
                <p className="text-sm text-neutral-600 dark:text-neutral-400 mb-4">{project.description}</p>
              </div>
              <div className="flex flex-wrap gap-2">
                {project.technologies.map((tech) => (
                  <span key={tech} className="text-xs font-mono bg-neutral-100 dark:bg-neutral-800 px-2 py-1 rounded">
                    {tech}
                  </span>
                ))}
              </div>
            </Link>
          ))}
        </div>
      </section>
    </div>
  );
}
